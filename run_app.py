import multiprocessing
import threading
import queue  
import uvicorn
import sys
import glob
from typing import List, Dict, Any

try:
    from workers.inference_worker import run_inference_worker
    from workers.camera_worker import run_camera_worker
    from workers.recording_worker import run_recording_worker 
    from api.main import app as fastapi_app
except ImportError as e:
    print(f"Error fatal: No se pudo importar un módulo. {e}")
    print("Asegúrate de ejecutar este script desde la raíz del proyecto (API_Model/).")
    sys.exit(1)

def orchestrator_listener(
    orchestrator_queue: multiprocessing.Queue,
    inference_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
    recording_queue: multiprocessing.Queue,
    video_frames_queue: multiprocessing.Queue,
    control_queues_dict: Dict[str, multiprocessing.Queue],
    worker_processes_dict: Dict[str, multiprocessing.Process]
):
    """
    Hilo en segundo plano (El Gerente) que escucha peticiones dinámicas de la API
    para instanciar o destruir procesos de cámaras sin apagar el sistema.
    """
    print("[Orquestador Dinámico] Listo y escuchando peticiones de red...")
    while True:
        try:
            msg = orchestrator_queue.get(timeout=1.0)
            
            if msg.get("command") == "SHUTDOWN":
                break

            action = msg.get("action")
            cam_id = msg.get("id")

            if action == "ADD":
                if cam_id in worker_processes_dict:
                    print(f"[Orquestador] Advertencia: La cámara {cam_id} ya está corriendo.")
                    continue

                print(f"[Orquestador] Levantando NUEVA cámara dinámicamente: {cam_id}...")
                
                new_control_queue = multiprocessing.Queue()
                control_queues_dict[cam_id] = new_control_queue

                worker = multiprocessing.Process(
                    target=run_camera_worker,
                    args=(
                        cam_id,
                        msg.get("type", "rtsp"),
                        msg.get("path"),
                        inference_queue,
                        new_control_queue,
                        video_frames_queue,
                        recording_queue,
                        results_queue
                    ),
                    daemon=True,
                )
                worker.start()
                
                worker_processes_dict[cam_id] = worker
                print(f"[Orquestador] Cámara {cam_id} en línea y transmitiendo.")

            elif action == "REMOVE":
                if cam_id in worker_processes_dict:
                    print(f"[Orquestador] Deteniendo cámara de forma segura: {cam_id}...")
                    worker = worker_processes_dict[cam_id]

                    worker.terminate()
                    worker.join(timeout=2)

                    del worker_processes_dict[cam_id]
                    if cam_id in control_queues_dict:
                        del control_queues_dict[cam_id]

                    results_queue.put({"type": "camera_removed", "camera_id": cam_id})
                    
                    inference_queue.put({"command": "REMOVE_CAMERA", "camera_id": cam_id})
                    
                    print(f"[Orquestador] Cámara {cam_id} apagada y memoria liberada.")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Orquestador Dinámico] Error inesperado: {e}")


def main(
    initial_cameras: List[Dict[str, Any]],
    inference_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
    recording_queue: multiprocessing.Queue,  
    orchestrator_queue: multiprocessing.Queue,
    video_frames_queue: multiprocessing.Queue
):
    """
    Punto de entrada principal.
    """
    print("=== Iniciando UrbanSentinel Backend V3 (Híbrido) ===")
    
    control_queues = {}
    worker_processes = {}

    try:
        fastapi_app.state.inference_queue = inference_queue
        fastapi_app.state.results_queue = results_queue
        fastapi_app.state.control_queues = control_queues
        fastapi_app.state.video_frames_queue = video_frames_queue
        fastapi_app.state.orchestrator_queue = orchestrator_queue 
        
        print("[Orquestador] Colas enlazadas a la memoria de la API.")
        
        print("[Orquestador] Levantando Motores Core (Inferencia y Grabación)...")
        inference_process = multiprocessing.Process(
            target=run_inference_worker, 
            args=(inference_queue, results_queue), 
            daemon=True
        )
        inference_process.start()

        recording_process = multiprocessing.Process(
            target=run_recording_worker, 
            args=(recording_queue, results_queue), 
            daemon=True
        )
        recording_process.start()

        listener_thread = threading.Thread(
            target=orchestrator_listener,
            args=(
                orchestrator_queue, inference_queue, results_queue, 
                recording_queue, video_frames_queue, control_queues, worker_processes
            ),
            daemon=True
        )
        listener_thread.start()

        for cam in initial_cameras:
            orchestrator_queue.put({
                "action": "ADD",
                "id": cam["id"],
                "type": cam["type"],
                "path": cam["path"]
            })

        print("\n=== API (FastAPI) Online: http://127.0.0.1:8010 ===")
        print("Presiona CTRL+C para apagar el sistema de forma segura.\n")
        
        uvicorn.run(
            fastapi_app,
            host="127.0.0.1",
            port=8010,
            log_level="info",
            reload=False, 
            loop="uvloop" if sys.platform != "win32" else "auto"
        )

    except KeyboardInterrupt:
        print("\n[Orquestador] Deteniendo servicios de forma segura...")
    finally:
        orchestrator_queue.put({"command": "SHUTDOWN"})
        
        if "inference_process" in locals() and inference_process.is_alive():
            inference_process.terminate()
            
        if "recording_process" in locals() and recording_process.is_alive():
            recording_process.terminate()
            
        for worker in list(worker_processes.values()):
            if worker.is_alive():
                worker.terminate()
                
        print("[Orquestador] Sistema apagado correctamente.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("--- Configurando Entorno de Producción / Pruebas ---")

    STARTUP_CAMERAS = []
    
    import urllib.request
    import json
    print("[Arranque] Consultando cámaras activas al Backend (Puerto 8000)...")
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/conexiones?habilitada=true")
        with urllib.request.urlopen(req, timeout=5) as response:
            camaras_bd = json.loads(response.read().decode())
            for cam in camaras_bd:
                if cam.get("rtsp_url") != "webcam" and cam.get("estado") != "eliminada":
                    STARTUP_CAMERAS.append({
                        "id": f"cam_{cam['id']:02d}", 
                        "type": "rtsp",
                        "path": cam["rtsp_url"]
                    })
        print(f"[Arranque] ¡Éxito! Se sincronizaron {len(STARTUP_CAMERAS)} cámaras desde la Base de Datos.")
    except Exception as e:
        print(f"[Arranque] Advertencia: No se pudo conectar con el Backend (8000). Iniciando vacío. Detalle: {e}")

    inference_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue() 
    orchestrator_queue = multiprocessing.Queue()
    video_frames_queue = multiprocessing.Queue(maxsize=32)
    
    print("[Orquestador] Tuberías IPC creadas con éxito.")

    main(
        STARTUP_CAMERAS,
        inference_queue,
        results_queue,
        recording_queue,
        orchestrator_queue,
        video_frames_queue
    )