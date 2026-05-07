import cv2
import threading
import time
import numpy as np
from typing import Tuple, Optional
from .base_reader import BaseReader

class RtspReader(BaseReader):
    """
    Lector optimizado para flujos de red (RTSP).
    Usa un hilo dedicado para evitar el retraso (lag) acumulado.
    """

    def __init__(self, source: str):
        self.rtsp_url = source
        self.cap = cv2.VideoCapture(self.rtsp_url)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_delay = 1.0 / self.fps

        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None
        self.is_connected = True
        self.running = True

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[RtspReader] Conectado a: {source} a {self.fps} FPS")

    def _update_loop(self):
        """Bucle en segundo plano que siempre mantiene el frame más reciente."""
        while self.running:
            if not self.is_connected:
                print("[RtspReader] Reconectando...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.is_connected = self.cap.isOpened()
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.last_frame = frame
                time.sleep(self.frame_delay * 0.5) 
            else:
                self.is_connected = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.last_frame is not None:
    
                return True, self.last_frame
            return False, None

    def get_fps(self) -> float:
        return self.fps

    def release(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()