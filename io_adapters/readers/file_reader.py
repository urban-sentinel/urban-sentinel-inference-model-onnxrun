import time
import av
import numpy as np
from typing import List, Union, Tuple, Optional
from .base_reader import BaseReader

class FileReader(BaseReader):
    """
    Lector optimizado con PyAV para pruebas locales. 
    Lee una lista de videos en bucle (looping infinito) con bajo uso de CPU.
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
        self.container = None
        self.video_stream = None
        self.frame_iter = None
        
        self.source_fps: float = 30.0
        self.last_frame_time = time.perf_counter()
        
        self._open_video(self.video_paths[self.current_video_index])

    def _open_video(self, file_path: str) -> bool:
        try:
            if self.container:
                self.container.close()
                
            self.current_file_path = file_path
            self.container = av.open(file_path)
            self.video_stream = self.container.streams.video[0]
            
            self.video_stream.thread_type = "AUTO" 
            
            self.frame_iter = self.container.decode(self.video_stream)
            
            if self.current_video_index == 0 and self.video_stream.average_rate:
                self.source_fps = float(self.video_stream.average_rate)
                
            return True
        except Exception as e:
            print(f"[FileReader] Error al abrir {file_path} con PyAV: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.container:
            return self._get_next_video()

        try:
            frame = next(self.frame_iter)
            img_array = frame.to_ndarray(format='bgr24')
            
            target_delay = 1.0 / self.source_fps
            elapsed = time.perf_counter() - self.last_frame_time
            sleep_time = target_delay - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time) 
                
            self.last_frame_time = time.perf_counter()
            return True, img_array
            
        except (StopIteration, av.EOFError):
            return self._get_next_video()
        except Exception as e:
            print(f"[FileReader] Error decodificando frame: {e}")
            return self._get_next_video()

    def _get_next_video(self) -> Tuple[bool, Optional[np.ndarray]]:
        self.current_video_index = (self.current_video_index + 1) % len(self.video_paths)
        new_path = self.video_paths[self.current_video_index]
        
        if not self._open_video(new_path):
            time.sleep(1) 
            return self._get_next_video() 
        
        return self.read()

    def get_fps(self) -> float:
        return self.source_fps

    def release(self) -> None:
        if self.container:
            self.container.close()