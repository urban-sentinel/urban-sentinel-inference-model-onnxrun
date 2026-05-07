import os

os.environ["YOLO_AUTOUPDATE"] = "false"

import sys
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from typing import Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config

class YoloSpatialTracker:
    """
    Motor espacial basado en YOLO11n (ONNX) y ByteTrack.
    Responsable de aislar las coordenadas de las personas en el frame.
    Diseñado para ser instanciado una vez por cada cámara activa.
    """
    
    def __init__(self, model_path: str = config.SPATIAL_MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Carga el modelo ONNX en la CPU aplicando límites estrictos de hilos
        para proteger la decodificación del video.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[YOLOSpatial] CRÍTICO: Modelo ONNX no encontrado en {self.model_path}. "
                "Por favor, ejecuta 'python scripts/export_yolo.py' primero para generarlo."
            )

        try:
           
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = config.YOLO_NUM_THREADS
            session_options.inter_op_num_threads = config.YOLO_NUM_THREADS
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.model = YOLO(self.model_path, task="detect")
            
            if hasattr(self.model, 'predictor') and self.model.predictor:
                self.model.predictor.session = ort.InferenceSession(
                    self.model_path, 
                    sess_options=session_options, 
                    providers=['CPUExecutionProvider']
                )
                
            print(f"[YOLOSpatial] Motor Espacial (ONNX-CPU) listo. (Hilos: {config.YOLO_NUM_THREADS})")

        except Exception as e:
            raise RuntimeError(f"[YOLOSpatial] Error fatal al cargar el modelo: {e}")

    def track_persons(self, frame: np.ndarray) -> Dict[int, List[int]]:
        """
        Evalúa el frame y rastrea temporalmente a los humanos detectados.
        
        Args:
            frame: np.ndarray de la cámara (ej. 1920x1080)
            
        Returns:
            Diccionario súper ligero con los IDs y coordenadas absolutas.
            Formato: { track_id: [x1, y1, x2, y2] }
        """
        if self.model is None:
            return {}

        try:
            
            results = self.model.track(
                source=frame,
                classes=[config.YOLO_PERSON_CLASS_ID],
                conf=config.YOLO_CONFIDENCE,
                tracker="bytetrack.yaml",
                persist=True,  
                verbose=False  
            )

            tracked_boxes = {}
            result = results[0]
            
            if result.boxes is None or result.boxes.id is None:
                return tracked_boxes

            boxes_xywh = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()

            for i, track_id in enumerate(track_ids):
                
                cx, cy, w, h = boxes_xywh[i]
                
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                
                tracked_boxes[int(track_id)] = [x1, y1, x2, y2]

            return tracked_boxes

        except Exception as e:
            print(f"[YOLOSpatial] Error durante el tracking: {e}")
            return {}