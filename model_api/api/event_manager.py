import asyncio
import json
import sys
import os
import numpy as np
from multiprocessing import Queue
from typing import Dict, Union

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 1 nivel: .../api -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from config import config
    from api.connection_manager import ConnectionManager
except ImportError as e:
    print(f"Error fatal en 'event_manager.py': No se pudo importar un módulo. {e}")
    sys.exit(1)

# Diccionario global para mantener el estado de cada cámara (ej. "IDLE", "RECORDING")# ... imports iguales ...

# Diccionario global para mantener el estado de cada cámara (ej. "IDLE", "RECORDING")
camera_states: Dict[str, str] = {}

async def event_manager_task(
    manager: ConnectionManager,
    results_queue: Queue,
    control_queues: Dict[str, Queue]
):
    print("[EventManager] Tarea de fondo iniciada. Esperando resultados de la GPU...")

    while True:
        try:
            item = await asyncio.to_thread(results_queue.get)

            # --- A) ¿Llegó un resumen de evento? (del worker al terminar STOP_RECORDING)
            if isinstance(item, dict) and item.get("type") == "event_complete":
                camera_id = item.get("camera_id")
                if camera_id:
                    # Reenviamos el resumen tal cual (incluye video_path y log_path)
                    await manager.broadcast(camera_id, json.dumps(item))
                continue

            # --- B) Si no es dict, asumimos el mensaje clásico de inferencia: (camera_id, probabilities)
            camera_id, probabilities = item

            # 1) Probabilidades y top-k
            probs_dict = {config.CLASSES[i]: float(probabilities[i]) for i in range(len(config.CLASSES))}
            top_idx = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
            top_prob = float(probabilities[top_idx])

            # 2) Umbral
            is_violence_detected = top_prob >= config.ALERT_THRESHOLD

            # 3) Broadcast SOLO si supera umbral (así no “spameamos”)
            if is_violence_detected:
                await manager.broadcast(camera_id, json.dumps({
                    "camera_id": camera_id,
                    "probabilities": probs_dict,
                    "triggered": True
                }))

            # 4) Máquina de estados (igual que ya tenías)
            current_state = camera_states.get(camera_id, "IDLE")
            control_queue = control_queues.get(camera_id)
            if not control_queue:
                print(f"[EventManager] ERROR: No se encontró 'control_queue' para {camera_id}.")
                continue

            if is_violence_detected:
                if current_state == "IDLE":
                    print(f"[EventManager] ¡Evento detectado en {camera_id}! Enviando orden START_RECORDING.")
                    control_queue.put("START_RECORDING")
                    camera_states[camera_id] = "RECORDING"
                # pasar probs al recorder si las usa
                control_queue.put(probabilities)

            elif current_state == "RECORDING":
                print(f"[EventManager] Evento terminado en {camera_id}. Enviando orden STOP_RECORDING.")
                control_queue.put("STOP_RECORDING")
                camera_states[camera_id] = "IDLE"

        except (KeyboardInterrupt, SystemExit):
            print("[EventManager] Deteniendo tarea de fondo...")
            break
        except Exception as e:
            print(f"[EventManager] ERROR en el bucle: {e}")
            await asyncio.sleep(1)