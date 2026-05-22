import sys
import os
import time
import queue
import cv2
import numpy as np
from multiprocessing import Queue
from queue import Empty

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config
from ai_engine.models.yolo_spatial import YoloSpatialTracker
from ai_engine.pipeline.track_buffer_manager import TrackBufferManager

def run_inference_worker(
    inference_queue: Queue, 
    results_queue: Queue
):
    """
    Proceso Consumidor Aislado (GPU).
    Orquestador Híbrido: Recolecta JPEGs, ejecuta YOLO (Batch), extrae Tubelets 
    y ejecuta DINOv3 (Batch) en un solo entorno de VRAM.
    """
    print("[InferenceWorker] Inicializando Motor Híbrido Espacio-Temporal en GPU...")

    try:
        import torch
        from ai_engine.models.dinov3_temporal import DinoV3TemporalPredictor
    except ImportError as e:
        print(f"[InferenceWorker] Error crítico de importación: {e}")
        return

    try:
        spatial_tracker = YoloSpatialTracker()
        temporal_predictor = DinoV3TemporalPredictor(model_path=config.TEMPORAL_MODEL_PATH)
        track_manager = TrackBufferManager()
    except Exception as e:
        print(f"[InferenceWorker] CRÍTICO: Fallo al cargar modelos de IA: {e}")
        return

    MAX_BATCH_SIZE = getattr(config, 'MAX_BATCH_SIZE', 8)
    BATCH_TIMEOUT = 0.04  
    
    print(f"[InferenceWorker] Listo. Operando a Max Batch: {MAX_BATCH_SIZE} | Timeout: {BATCH_TIMEOUT*1000}ms")

    while True:
        try:
            batch_items = []
            start_time = time.time()

            while len(batch_items) < MAX_BATCH_SIZE:
                time_elapsed = time.time() - start_time
                time_left = BATCH_TIMEOUT - time_elapsed
                
                if time_left <= 0:
                    break
                    
                try:
                    msg = inference_queue.get(timeout=time_left)
                    
                    if isinstance(msg, dict) and msg.get("command") == "REMOVE_CAMERA":
                        cam_to_remove = msg.get("camera_id")
                        spatial_tracker.remove_camera(cam_to_remove)
                        track_manager.remove_camera(cam_to_remove)
                        continue
                        
                    batch_items.append(msg)
                except Empty:
                    break

            if not batch_items:
                continue

            frames = []
            cam_ids = []
            frame_idxs = []

            for item in batch_items:
                cam_ids.append(item["camera_id"])
                frame_idxs.append(item["frame_idx"])
                
                img_array = np.frombuffer(item["jpeg_bytes"], np.uint8)
                frames.append(cv2.imdecode(img_array, cv2.IMREAD_COLOR))

            batch_results = spatial_tracker.track_batch(frames, cam_ids)

            all_ready_tubelets = []
            tubelets_metadata = []

            for i, cam_id in enumerate(cam_ids):
                raw_boxes = batch_results.get(cam_id, {})
                frame_img = frames[i]
                f_idx = frame_idxs[i]

                ready_tubelets, final_boxes = track_manager.process_frame(cam_id, frame_img, raw_boxes, f_idx)

                if final_boxes:
                    results_queue.put((cam_id, {"type": "boxes", "data": final_boxes}))

                for t in ready_tubelets:
                    all_ready_tubelets.append(t["tubelet_data"])
                    tubelets_metadata.append({"cam_id": cam_id, "group_id": t["track_id"]})

            if all_ready_tubelets:
                start_infer = time.time()
                
                tensor_batch = torch.tensor(np.stack(all_ready_tubelets), dtype=torch.float32)
                tensor_batch = tensor_batch.permute(0, 4, 1, 2, 3)
                
                batch_probs = temporal_predictor.predict_batch(tensor_batch)
                infer_time = (time.time() - start_infer) * 1000

                final_results_by_cam = {}
                
                for idx, meta in enumerate(tubelets_metadata):
                    c_id = meta["cam_id"]
                    g_id = meta["group_id"]
                    probs = batch_probs[idx].tolist()
                    clase_dominante = config.CLASSES[np.argmax(probs)]
                    
                    if c_id not in final_results_by_cam:
                        final_results_by_cam[c_id] = {}
                        
                    final_results_by_cam[c_id][g_id] = {
                        "clase_dominante": clase_dominante,
                        "probabilidades": probs
                    }
                    
                    probs_str = ", ".join([f"{c}: {p:.2f}" for c, p in zip(config.CLASSES, probs)])
                    print(f"[Inferencia] {c_id} | Zona: {g_id} | Dominante: {clase_dominante} | Detalle: [{probs_str}]")
                
                for c_id, res_data in final_results_by_cam.items():
                    results_queue.put((c_id, res_data))
                    
                print(f"[InferenceWorker] {len(tubelets_metadata)} Tubelets procesados en {infer_time:.1f}ms")

        except KeyboardInterrupt:
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[InferenceWorker] ¡OOM! Liberando caché de la GPU...")
                torch.cuda.empty_cache()
            else:
                print(f"[InferenceWorker] Error de PyTorch: {e}")
        except Exception as e:
            print(f"[InferenceWorker] Error inesperado en el ciclo central: {e}")

    print("[InferenceWorker] Proceso finalizado.")