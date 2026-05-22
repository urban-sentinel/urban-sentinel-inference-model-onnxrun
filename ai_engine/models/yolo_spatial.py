import os

os.environ["YOLO_AUTOUPDATE"] = "false"

import sys
import numpy as np
import torch
import ultralytics.utils.checks as checks
checks.check_requirements = lambda *args, **kwargs: None
from ultralytics import YOLO
from typing import Dict, List, Optional
from types import SimpleNamespace

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except ImportError:
    BYTETracker = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config

class YoloSpatialTracker:
    """
    Motor espacial basado en YOLO11n (PyTorch).
    Arquitectura Desacoplada: Procesamiento en Batch (GPU) y Rastreo Aislado (CPU).
    """
    
    def __init__(self, model_path: str = config.SPATIAL_MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self.trackers: Dict[str, BYTETracker] = {}
        
        self.tracker_args = SimpleNamespace(
            tracker_type='bytetrack',
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=config.TRACKER_MAX_AGE,
            match_thresh=0.8,
            mot20=False,
            fuse_score=True,             
            gmc_method='sparseOptFlow'   
        )
        
        self._initialize_model()

    def _initialize_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[YOLOSpatial] CRÍTICO: Modelo no encontrado en {self.model_path}.")

        try:
            print(f"[YOLOSpatial] Cargando modelo desde: {self.model_path}")
            self.model = YOLO(self.model_path, task="detect")
            
            print("[YOLOSpatial] Ejecutando calentamiento de motor (Warmup en GPU)...")
            dummy_batch = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2
            
            self.model.predict(
                source=dummy_batch, 
                classes=[config.YOLO_PERSON_CLASS_ID], 
                verbose=False, 
                device="cuda" 
            )
            print(f"[YOLOSpatial] Motor Espacial cargado y atado a la GPU (CUDA).")

        except Exception as e:
            raise RuntimeError(f"[YOLOSpatial] Error fatal al cargar el modelo: {e}")

    def _get_tracker(self, camera_id: str) -> BYTETracker:
        if camera_id not in self.trackers:
            if BYTETracker is None:
                raise ImportError("No se pudo importar BYTETracker de ultralytics.")
            self.trackers[camera_id] = BYTETracker(self.tracker_args, frame_rate=config.TARGET_FPS)
        return self.trackers[camera_id]

    def track_batch(self, batch_frames: List[np.ndarray], camera_ids: List[str]) -> Dict[str, Dict[int, List[int]]]:
        if self.model is None or not batch_frames:
            return {cam_id: {} for cam_id in camera_ids}

        batch_results = {}

        try:
           
            results = self.model.predict(
                source=batch_frames,
                classes=[config.YOLO_PERSON_CLASS_ID],
                conf=config.YOLO_CONFIDENCE,
                verbose=False,
                device="cuda" 
            )

            for i, result in enumerate(results):
                cam_id = camera_ids[i]
                tracked_boxes_cam = {}

                if result.boxes is None or len(result.boxes) == 0:
                    batch_results[cam_id] = tracked_boxes_cam
                    continue

                boxes_on_cpu = result.boxes.cpu()

                tracker = self._get_tracker(cam_id)
                tracks = tracker.update(boxes_on_cpu, img=batch_frames[i])

                for track in tracks:
                
                    x1, y1, x2, y2 = map(int, track[:4])
                    
                    track_id = int(track[4])
                    
                    tracked_boxes_cam[track_id] = [x1, y1, x2, y2]

                batch_results[cam_id] = tracked_boxes_cam

            return batch_results

        except Exception as e:
            import traceback
            print(f"[YOLOSpatial] Error CRÍTICO durante el tracking en batch:\n{traceback.format_exc()}")
            return {cam_id: {} for cam_id in camera_ids}
        
    def remove_camera(self, camera_id: str):
        if camera_id in self.trackers:
            del self.trackers[camera_id]