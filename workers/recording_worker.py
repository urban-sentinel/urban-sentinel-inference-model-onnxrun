import sys
import os
import time
from multiprocessing import Queue
from queue import Empty
from typing import Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.config import config
from io_adapters.writers.base_writer import BaseWriter
from io_adapters.writers.disk_recorder import DiskRecorder

def run_recording_worker(recording_queue: Queue, results_queue: Queue):
    """
    Proceso Aislado (CPU). Se dedica 100% a escuchar la cola de grabación y 
    escribir en disco duro usando DiskRecorder, sin bloquear a las cámaras.
    """

    print("[RecordingWorker] Proceso de grabación iniciado y en espera.")
    
    active_recorders: Dict[str, BaseWriter] = {}

    try:
        while True:
            try:
                
                message = recording_queue.get(timeout=0.1)
                
                command = message[0]
                camera_id = message[1]

                # INICIAR GRABACIÓN 
                if command == "START":
                    pre_roll_frames = message[2]
                    source_fps = message[3]
                    
                    if camera_id not in active_recorders:
                       
                        recorder = DiskRecorder(
                            camera_id=camera_id, 
                            source_fps=source_fps, 
                            video_dir=config.SAVE_CLIP_PATH, 
                            log_dir=config.SAVE_LOG_PATH
                        )
                        success = recorder.start_recording(pre_roll_frames)
                        
                        if success:
                            active_recorders[camera_id] = recorder
                            print(f"[RecordingWorker] Iniciando grabación para {camera_id}")

                # AGREGAR FRAME 
                elif command == "FRAME":
                    frame = message[2]
                    metadata = message[3]
                    
                    if camera_id in active_recorders:
                        active_recorders[camera_id].write_frame(frame, metadata)

                # DETENER GRABACIÓN 
                elif command == "STOP":
                    if camera_id in active_recorders:
                        recorder = active_recorders.pop(camera_id)
                        summary = recorder.close()
                
                        if summary and results_queue is not None:
                            results_queue.put({"type": "event_complete", **summary})
                            
                # APAGAR WORKER 
                elif command == "SHUTDOWN":
                    print("[RecordingWorker] Recibida señal de apagado. Cerrando grabaciones activas...")
                    break

            except Empty:
                continue

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[RecordingWorker] ERROR CRÍTICO: {e}")
    finally:
        for cam_id, recorder in active_recorders.items():
            print(f"[RecordingWorker] Cerrando grabación de emergencia para {cam_id}")
            recorder.close()
        print("[RecordingWorker] Proceso finalizado.")