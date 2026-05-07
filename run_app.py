import multiprocessing
import uvicorn
import sys
import time
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

def main(
    cameras_to_run: List[Dict[str, Any]],
    inference_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
    recording_queue: multiprocessing.Queue,  
    control_queues: Dict[str, multiprocessing.Queue],
    video_frames_queue: multiprocessing.Queue,
):
    """
    Orquesta los servicios. Levanta los motores de IA y la API web.
    """
    print("=== Iniciando UrbanSentinel Backend V2 ===")
    worker_processes = []

    try:

        fastapi_app.state.inference_queue = inference_queue
        fastapi_app.state.results_queue = results_queue
        fastapi_app.state.control_queues = control_queues
        fastapi_app.state.video_frames_queue = video_frames_queue
        print("[Orquestador] Colas enlazadas a la memoria de la API.")
        print("[Orquestador] Levantando Motor del modelo...")
        inference_process = multiprocessing.Process(
            target=run_inference_worker,
            args=(inference_queue, results_queue),
            daemon=True,
        )
        inference_process.start()

        print("[Orquestador] Levantando Motor de Grabación (I/O Aislado)...")
        recording_process = multiprocessing.Process(
            target=run_recording_worker,
            args=(recording_queue, results_queue),
            daemon=True,
        )
        recording_process.start()

        for cam in cameras_to_run:
            print(f"[Orquestador] Levantando cámara: {cam['id']}...")
            worker = multiprocessing.Process(
                target=run_camera_worker,
                args=(
                    cam["id"],
                    cam["type"],
                    cam["path"],
                    inference_queue,
                    control_queues[cam["id"]],
                    video_frames_queue,
                    recording_queue,  
                    results_queue
                ),
                daemon=True,
            )
            worker.start()
            worker_processes.append(worker)

        print(f"[Orquestador] {len(worker_processes)} procesos de ingesta corriendo.")
        print("\n=== API (FastAPI) Online: http://127.0.0.1:8010 ===")
        print("Presiona CTRL+C para apagar el sistema de forma segura.\n")
        
        uvicorn.run(
            fastapi_app,
            host="127.0.0.1",
            port=8010,
            log_level="info",
            reload=False, 
        )

    except KeyboardInterrupt:
        print("\n[Orquestador] Deteniendo servicios de forma segura...")
    finally:
        if "inference_process" in locals() and inference_process.is_alive():
            inference_process.terminate()
            
        if "recording_process" in locals() and recording_process.is_alive():
            recording_process.terminate()
            
        for worker in worker_processes:
            if worker.is_alive():
                worker.terminate()
                
        print("[Orquestador] Sistema apagado correctamente.")


if __name__ == "__main__":
   
    multiprocessing.set_start_method("spawn")

    print("--- Configurando Entorno de Producción / Pruebas ---")

    videos_de_prueba = glob.glob("test_videos/*.mp4") 

    CAMERAS_TO_RUN = [
        {"id": "cam_simulada_01", "type": "file", "path": videos_de_prueba} 
    ]

    inference_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    recording_queue = multiprocessing.Queue() 
    
    control_queues = {cam["id"]: multiprocessing.Queue() for cam in CAMERAS_TO_RUN}
    
    video_frames_queue = multiprocessing.Queue(maxsize=32)
    
    print("[Orquestador] Tuberías de memoria IPC creadas con éxito.")

    main(
        CAMERAS_TO_RUN,
        inference_queue,
        results_queue,
        recording_queue,
        control_queues,
        video_frames_queue,
    )