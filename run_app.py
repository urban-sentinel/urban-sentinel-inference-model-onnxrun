import multiprocessing
import uvicorn
import os
import sys
import time
import glob
import random
import numpy as np
from typing import List, Dict, Any

# --- 1. Importar los Componentes del Backend ---
try:
    from model_api.services.inference_service import run_inference_service
    from model_api.services.camera_worker import run_camera_worker
    from model_api.api import main as api_main
    from model_api.config import config
except ImportError as e:
    print(f"Error fatal: No se pudo importar un módulo desde 'model_api'. {e}")
    print("Asegúrate de que 'run_app.py' esté en la raíz del proyecto (junto a 'model_api').")
    sys.exit(1)

def main(
    cameras_to_run: List[Dict[str, Any]],
    inference_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
    control_queues: Dict[str, multiprocessing.Queue],
    video_frames_queue: multiprocessing.Queue,  # <<< NUEVO
):
    """
    Orquesta los servicios. Recibe la configuración y las colas desde __main__.
    """
    print("--- Iniciando UrbanSentinel Backend ---")
    worker_processes = []

    try:
        # --- 1) Inyectar colas en la API ---
        api_main.inference_queue = inference_queue
        api_main.results_queue = results_queue
        api_main.control_queues = control_queues
        api_main.video_frames_queue = video_frames_queue  # <<< NUEVO
        print("Colas inyectadas en el módulo API (incluida video_frames_queue).")

        # --- 2) Servicio de inferencia ---
        print("Iniciando servicio de inferencia (Proceso GPU/CPU)...")
        inference_process = multiprocessing.Process(
            target=run_inference_service,
            args=(inference_queue, results_queue),
            daemon=True,
        )
        inference_process.start()

        # --- 3) Workers de Cámara ---
        for cam in cameras_to_run:
            print(f"Iniciando worker para cámara: {cam['id']}...")
            worker = multiprocessing.Process(
                target=run_camera_worker,
                args=(
                    cam["id"],
                    cam["type"],
                    cam["path"],
                    inference_queue,
                    control_queues[cam["id"]],
                    video_frames_queue,  # <<< NUEVO: cola para enviar JPEGs al API
                    results_queue
                ),
                daemon=True,
            )
            worker.start()
            worker_processes.append(worker)

        print(f"{len(worker_processes)} workers de cámara iniciados.")

        # --- 4) API ---
        print("\n--- Iniciando API (FastAPI) en http://127.0.0.1:8010 ---")
        print("Puedes iniciar 'test_websocket.py' en otra terminal para ver los resultados.")
        uvicorn.run(
            "model_api.api.main:app",
            host="127.0.0.1",
            port=8010,
            log_level="info",
            reload=False,
        )

    except KeyboardInterrupt:
        print("\nDeteniendo servicios...")
    finally:
        print("Enviando señal de terminación a los procesos...")
        if "inference_process" in locals() and inference_process.is_alive():
            inference_process.terminate()
        for worker in worker_processes:
            if worker.is_alive():
                worker.terminate()
        print("Servicios detenidos. Saliendo.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    print("--- Configurando la prueba de 1 cámara web ---")

    CAMERAS_TO_RUN = [
        {"id": "cam_01", "type": "webcam", "path": 0}  # "0" = primera webcam
    ]

    # 3) Crear Colas
    inference_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    control_queues = {cam["id"]: multiprocessing.Queue() for cam in CAMERAS_TO_RUN}
    video_frames_queue = multiprocessing.Queue(maxsize=32)  # <<< NUEVO (ligera y acotada)
    print("Colas de comunicación creadas (incluida video_frames_queue).")

    # 4) Lanzar orquestador
    main(
        CAMERAS_TO_RUN,
        inference_queue,
        results_queue,
        control_queues,
        video_frames_queue,  # <<< NUEVO
    )
