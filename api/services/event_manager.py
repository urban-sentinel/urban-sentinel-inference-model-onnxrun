import asyncio
import orjson
import sys
import os
import time  
from multiprocessing import Queue
from queue import Empty
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config
from api.services.connection_manager import ConnectionManager
from api.services.prediction_filter import PredictionFilter  

camera_recording_state: Dict[str, bool] = {}
camera_event_memory: Dict[str, Dict[str, List[float]]] = {}
camera_last_violence_time: Dict[str, float] = {}  

prediction_filter = PredictionFilter()

async def event_manager_task(
    manager: ConnectionManager,
    results_queue: Queue,
    control_queues: Dict[str, Queue]
):
    print("[EventManager] Enlace API-WebSocket iniciado. Escuchando resultados de GPU...")

    while True:
        try:
            item = await asyncio.to_thread(results_queue.get, True, 1.0)

            if isinstance(item, dict) and item.get("type") == "event_complete":
                camera_id = item.get("camera_id")
                if camera_id:
                    log_path = item.get("log_path")
                    memory = camera_event_memory.pop(camera_id, None)
                    
                    if log_path and os.path.exists(log_path) and memory:
                        try:
                            averages = {k: round(sum(v)/len(v), 4) for k, v in memory.items() if v}
                            
                            with open(log_path, 'rb') as f:
                                log_data = orjson.loads(f.read())
                            
                            log_data["analisis_ia"] = {
                                "metricas_promedio": averages
                            }
                            
                            with open(log_path, 'wb') as f:
                                f.write(orjson.dumps(log_data, option=orjson.OPT_INDENT_2))
                        except Exception as e:
                            print(f"[EventManager] Error inyectando métricas en JSON: {e}")

                    print(f"[EventManager] Grabación lista para {camera_id}: {item.get('video_path')}")
                    await manager.broadcast(camera_id, orjson.dumps(item).decode('utf-8'))
                continue

            if isinstance(item, dict) and item.get("type") == "camera_removed":
                camera_id = item.get("camera_id")
                if camera_id:
                    print(f"[EventManager] Limpiando memoria y registros de cámara eliminada: {camera_id}")
                    camera_recording_state.pop(camera_id, None)
                    camera_event_memory.pop(camera_id, None)
                    camera_last_violence_time.pop(camera_id, None)
                    prediction_filter.clear_camera(camera_id)
                continue

            if isinstance(item, tuple) and isinstance(item[1], dict) and item[1].get("status") == "clear":
                camera_id = item[0]
                prediction_filter.clear_camera(camera_id)  
                
                if camera_recording_state.get(camera_id, False):
                    last_violence = camera_last_violence_time.get(camera_id, 0)
                    cooldown = getattr(config, 'COOLDOWN_SECONDS', 5)
                    
                    if time.time() - last_violence >= cooldown:
                        control_queue = control_queues.get(camera_id)
                        if control_queue:
                            print(f"[EventManager] Calle vacía y Cooldown finalizado en {camera_id}. Deteniendo grabación.")
                            control_queue.put({"command": "STOP_RECORDING"})
                            camera_recording_state[camera_id] = False
                            camera_last_violence_time[camera_id] = 0
                        
                await manager.broadcast(camera_id, orjson.dumps({
                    "camera_id": camera_id,
                    "status": "clear",
                    "triggered": False
                }).decode('utf-8'))
                continue

            if isinstance(item, tuple) and len(item) == 2:
                camera_id = item[0]
                inference_data = item[1]
                
                control_queue = control_queues.get(camera_id)
                if not control_queue:
                    continue

                any_violence_in_camera = False
                violent_groups = []

                is_currently_recording = camera_recording_state.get(camera_id, False)

                for group_id, data in inference_data.items():
                    probs = data["probabilidades"]
                    clase_dom_original = data["clase_dominante"]
                    
                    probs_dict = {config.CLASSES[i]: probs[i] for i in range(len(config.CLASSES))}
                    
                    clases_violentas = ["Golpe", "Patada", "Forcejeo"]
                    
                    clase_violenta_dom = max(clases_violentas, key=lambda k: probs_dict.get(k, 0.0))
                    max_violent_prob = probs_dict.get(clase_violenta_dom, 0.0)

                    is_violent, smoothed_prob = prediction_filter.update_and_check(
                        camera_id, group_id, max_violent_prob, clase_violenta_dom
                    )

                    if is_violent:
                        any_violence_in_camera = True
                        
                        clase_final_reportada = clase_violenta_dom
                        
                        violent_groups.append({
                            "group_id": group_id,
                            "accion": clase_final_reportada,
                            "probabilidad_suavizada": smoothed_prob
                        })

                        await manager.broadcast(camera_id, orjson.dumps({
                            "camera_id": camera_id,
                            "group_id": group_id,
                            "probabilities": probs_dict,
                            "clase_dominante": clase_final_reportada,
                            "smoothed_prob": smoothed_prob,  
                            "triggered": True
                        }).decode('utf-8'))

                    if any_violence_in_camera or is_currently_recording:
                        if camera_id not in camera_event_memory:
                            camera_event_memory[camera_id] = {cls: [] for cls in config.CLASSES}
                        for cls, val in probs_dict.items():
                            camera_event_memory[camera_id][cls].append(val)

                if any_violence_in_camera:
                    camera_last_violence_time[camera_id] = time.time()
                    
                    if not is_currently_recording:
                        print(f"[EventManager] PELEA DETECTADA (Filtro Superado) en {camera_id}: {violent_groups}. Iniciando grabación...")
                        control_queue.put({"command": "START_RECORDING"})
                        camera_recording_state[camera_id] = True
                else:
                    if is_currently_recording:
                        last_violence = camera_last_violence_time.get(camera_id, 0)
                        cooldown = getattr(config, 'COOLDOWN_SECONDS', 5)
                        
                        if time.time() - last_violence >= cooldown:
                            print(f"[EventManager] Escena pacífica consolidada. Deteniendo grabación en {camera_id}.")
                            control_queue.put({"command": "STOP_RECORDING"})
                            camera_recording_state[camera_id] = False
                            camera_last_violence_time[camera_id] = 0
                            
                            await manager.broadcast(camera_id, orjson.dumps({
                                "camera_id": camera_id,
                                "status": "safe",
                                "triggered": False
                            }).decode('utf-8'))

        except Empty:
            await asyncio.sleep(0.01)
        except (KeyboardInterrupt, SystemExit):
            print("[EventManager] Apagando enlace WebSocket...")
            break
        except Exception as e:
            print(f"[EventManager] ERROR CRÍTICO en el bucle principal: {e}")
            await asyncio.sleep(1)