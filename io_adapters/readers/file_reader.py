import time
import cv2
import os
import numpy as np
from typing import List, Union, Tuple, Optional
from .base_reader import BaseReader

class FileReader(BaseReader):
    """
    Lector para pruebas locales. Lee una lista de videos en bucle (looping infinito).
    """
    
    def __init__(self, source: Union[str, List[str]]):
        if isinstance(source, str):
            self.video_paths = [source]
        elif isinstance(source, list):
            self.video_paths = source
        else:
            raise TypeError("FileReader 'source' debe ser str o List[str]")

        if not self.video_paths:
            raise ValueError("FileReader 'source' no puede ser una lista vacía.")

        self.current_video_index = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_fps: float = 30.0
        self.last_frame_time = time.perf_counter()
        
        self._open_video(self.video_paths[self.current_video_index])

    def _open_video(self, file_path: str) -> bool:
        self.current_file_path = file_path
        self.cap = cv2.VideoCapture(self.current_file_path)
        
        if not self.cap.isOpened():
            print(f"[FileReader] Error al abrir: {file_path}")
            return False
            
        if self.current_video_index == 0:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.source_fps = fps
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap or not self.cap.isOpened():
            return self._get_next_video()

        ret, frame = self.cap.read()
        
        if not ret:
            return self._get_next_video()
        
        target_delay = 1.0 / self.source_fps
        elapsed = time.perf_counter() - self.last_frame_time
        sleep_time = target_delay - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time) 
            
        self.last_frame_time = time.perf_counter()
            
        return ret, frame

    def _get_next_video(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap:
            self.cap.release()
            
        self.current_video_index = (self.current_video_index + 1) % len(self.video_paths)
        new_path = self.video_paths[self.current_video_index]
        
        if not self._open_video(new_path):
            return self._get_next_video() 
        
        return self.cap.read()

    def get_fps(self) -> float:
        return self.source_fps

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()