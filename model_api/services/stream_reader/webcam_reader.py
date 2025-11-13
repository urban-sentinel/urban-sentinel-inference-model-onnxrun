import cv2
import time
import os
import sys
import numpy as np
from typing import Optional, Tuple, Any # <-- 1. Importar Tuple y Any

# Agregamos la raíz del proyecto ('model_api') al path de Python
# Sube 3 niveles: .../stream_reader -> .../services -> .../model_api
model_api_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(model_api_root)

try:
    # Usamos importación relativa (el punto) para 'base_reader'
    from .base_reader import BaseReader
except ImportError:
    # Fallback si la importación relativa falla (ej. al ejecutar como script)
    from services.stream_reader.base_reader import BaseReader

class WebcamReader(BaseReader):
    
    # 2. CAMBIO: __init__ debe aceptar 'source' (para cumplir con BaseReader)
    #    'source' será el índice del dispositivo, ej: "0"
    def __init__(self, source: Any, width: int = 640, height: int = 480, target_fps: int = 30):
        
        try:
            device_index = int(source)
        except ValueError:
            print(f"[WebcamReader] ERROR: 'source' debe ser un índice numérico (ej. 0), pero se recibió {source}")
            raise
            
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)  # CAP_DSHOW en Windows
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la webcam (índice {device_index})")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,         target_fps)
        
        self._target_fps = target_fps
        self._next_t = time.monotonic()
        print(f"[WebcamReader] Webcam (índice {device_index}) abierta.")

    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        return fps if fps > 0 else float(self._target_fps)

    # 3. CAMBIO: El método read() ahora devuelve el tuple (bool, frame)
    def read(self) -> tuple[bool, np.ndarray | None]:
        # pequeño regulador a ~target_fps
        now = time.monotonic()
        delay = self._next_t - now
        if delay > 0: 
            time.sleep(delay)
        self._next_t = max(now, self._next_t) + 1.0 / float(self._target_fps)

        ok, frame = self.cap.read()
        
        if not ok:
            print("[WebcamReader] Error al leer frame de la webcam.")
            return False, None # <-- Devolver (False, None)
            
        return True, frame  # <-- Devolver (True, frame)

    def release(self) -> None:
        try: 
            self.cap.release()
            print("[WebcamReader] Webcam liberada.")
        except: 
            pass