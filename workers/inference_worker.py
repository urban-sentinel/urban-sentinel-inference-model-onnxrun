import sys
import os
import time
from multiprocessing import Queue, shared_memory
from queue import Empty
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def run_inference_worker(inference_queue: Queue, results_queue: Queue, shared_manager):
    """
    Proceso Consumidor Aislado (GPU).
    Recibe los 'tickets' de Memoria Compartida, reconstruye los tensores,
    arma el Batch y ejecuta la inferencia de violencia.
    """
    try:
        import torch
        from ai_engine.models.dinov3_temporal import DinoV3TemporalPredictor
        from core.config import config
    except ImportError as e:
        print(f"[InferenceWorker] Error crítico de importación: {e}")
        return

    print("[InferenceWorker] Inicializando Motor Temporal en GPU...")

    try:
        predictor = DinoV3TemporalPredictor(model_path=config.TEMPORAL_MODEL_PATH)
    except Exception as e:
        print(f"[InferenceWorker] CRÍTICO: No se pudo cargar DINOv3: {e}")
        return

    print("[InferenceWorker] Listo y escuchando secuencias de video (Zero-Copy).")

    while True:
        try:
            message = inference_queue.get(timeout=0.1)
            camera_id = message[0]
            tubelets_list = message[1]

            if not tubelets_list:
                continue

            track_ids = []
            tensores_individuales = []
            indices_a_liberar = []

            try:
                for item in tubelets_list:
                    track_ids.append(item["track_id"])
                    
                    shm_index = item["shm_index"]
                    shm_name = item["shm_name"]
                    indices_a_liberar.append(shm_index)

                    existing_shm = shared_memory.SharedMemory(name=shm_name)
                    
                    shm_array = np.ndarray(
                        config.SHM_TENSOR_SHAPE, 
                        dtype=config.SHM_TENSOR_DTYPE, 
                        buffer=existing_shm.buf
                    )

                    tensor = torch.from_numpy(shm_array.copy())
                    tensores_individuales.append(tensor)

                    existing_shm.close()

                batch_tensor = torch.stack(tensores_individuales, dim=0)

                start_infer = time.time()
                batch_probs = predictor.predict_batch(batch_tensor)
                infer_time = (time.time() - start_infer) * 1000

                results_data = {}
                for i, group_id in enumerate(track_ids):
                    probabilidades = batch_probs[i].tolist() 
                    clase_dominante = config.CLASSES[np.argmax(probabilidades)]
                    
                    results_data[group_id] = {
                        "clase_dominante": clase_dominante,
                        "probabilidades": probabilidades
                    }

                results_queue.put((camera_id, results_data))
                print(f"[InferenceWorker] {camera_id}: {len(track_ids)} Tubelets procesados en {infer_time:.1f}ms")

            finally:
                for idx in indices_a_liberar:
                    shared_manager.release_block(idx)

        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[InferenceWorker] ¡OOM! Liberando caché de la GPU...")
                torch.cuda.empty_cache()
            else:
                print(f"[InferenceWorker] Error de PyTorch: {e}")
        except Exception as e:
            print(f"[InferenceWorker] Error inesperado procesando clip de {camera_id}: {e}")

    print("[InferenceWorker] Proceso finalizado.")