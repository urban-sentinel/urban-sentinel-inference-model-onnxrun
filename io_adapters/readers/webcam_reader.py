import cv2
import numpy as np
from typing import Any, Tuple, Optional
from .base_reader import BaseReader

class WebcamReader(BaseReader):
    """Lector directo de dispositivos físicos (Webcams)."""
    
    def __init__(self, source: Any, width: int = 640, height: int = 480, target_fps: int = 30):
        try:
            device_index = int(source)
        except ValueError:
            raise ValueError(f"[WebcamReader] 'source' debe ser un int, no {source}")
            
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)  
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam {device_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        self._target_fps = target_fps
        print(f"[WebcamReader] Webcam {device_index} abierta.")

    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        return fps if fps > 0 else float(self._target_fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()