import av
import numpy as np
import os
import json
import time
import cv2
from datetime import datetime
from typing import List, Dict, Optional
from .base_writer import BaseWriter

class DiskRecorder(BaseWriter):
    """
    Escritor asíncrono y ligero. Escribe el frame al disco usando PyAV.
    Minimiza el uso de CPU utilizando el preset 'ultrafast'.
    """

    def __init__(self, camera_id: str, source_fps: float, video_dir: str, log_dir: str):
        self.camera_id = camera_id
    
        self.source_fps = int(round(source_fps)) if source_fps > 0 else 30
        self.video_dir = video_dir
        self.log_dir = log_dir
        
        self.is_open = False
        self.container = None
        self.stream = None
        
        self.video_path = ""
        self.log_path = ""
        self.start_time = 0.0
        self.logs: List[Dict] = []

    def start_recording(self, pre_roll_frames: List[np.ndarray]) -> bool:
        if not pre_roll_frames:
            print(f"[DiskRecorder-{self.camera_id}] ERROR: No hay frames de pre-rollo.")
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_basename = f"{self.camera_id}_{timestamp}"
        
        self.video_path = os.path.join(self.video_dir, f"{file_basename}.mp4")
        self.log_path = os.path.join(self.log_dir, f"{file_basename}.json")

        try:
            h, w, _ = pre_roll_frames[0].shape
            
            self.container = av.open(self.video_path, mode='w')
            self.stream = self.container.add_stream('libx264', rate=self.source_fps)
            
            self.stream.width = w
            self.stream.height = h
            self.stream.pix_fmt = 'yuv420p'
            
            self.stream.options = {'preset': 'ultrafast', 'crf': '28'}

            self.is_open = True
            self.start_time = time.time()
            self.logs = []
            
            print(f"[DiskRecorder-{self.camera_id}] Grabación iniciada (PyAV): {file_basename}.mp4")
            
            for frame in pre_roll_frames:
                self._mux_frame(frame)
                
            return True

        except Exception as e:
            print(f"[DiskRecorder-{self.camera_id}] ERROR Crítico al iniciar: {e}")
            self.is_open = False
            if self.container:
                self.container.close()
            return False

    def _mux_frame(self, frame_array: np.ndarray):
        """Convierte Numpy a VideoFrame y lo empaqueta."""
        frame_av = av.VideoFrame.from_ndarray(frame_array, format='bgr24')
        for packet in self.stream.encode(frame_av):
            self.container.mux(packet)

    def write_frame(self, frame: np.ndarray, metadata: Dict[str, float]) -> None:
        """
        Dibuja las Super Cajas sobre el frame (solo para grabación) 
        y empaqueta el video junto con el registro JSON.
        """
        if not self.is_open or self.container is None:
            return

        try:
     
            frame_to_record = frame.copy()
            
            if isinstance(metadata, dict):
                for group_id, box in metadata.items():
                    
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    cv2.rectangle(frame_to_record, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    cv2.putText(
                        frame_to_record, 
                        f"ZONA: {group_id}", 
                        (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 0, 255), 
                        2
                    )
    
            self._mux_frame(frame_to_record)
            
            log_entry = {
                "timestamp_ms": int((time.time() - self.start_time) * 1000),
                "metadata": metadata 
            }
            self.logs.append(log_entry)
            
        except Exception as e:
            print(f"[DiskRecorder-{self.camera_id}] Error al escribir frame: {e}")

    def close(self) -> Optional[Dict]:
        """
        Drena los últimos paquetes, cierra recursos y guarda el JSON.
        """
        if not self.is_open:
            return None
            
        print(f"[DiskRecorder-{self.camera_id}] Cerrando archivo: {os.path.basename(self.video_path)}...")
        
        try:
            self.is_open = False
            
            if self.container and self.stream:
                for packet in self.stream.encode():
                    self.container.mux(packet)
                self.container.close()
            
            summary = {
                "camera_id": self.camera_id,
                "event_start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "event_end_time": datetime.now().isoformat(),
                "video_file": os.path.basename(self.video_path),
                "log_file": os.path.basename(self.log_path),
                "video_path": self.video_path,
                "log_path": self.log_path,
                "total_logs": len(self.logs),
                "logs": self.logs
            }
            
            with open(self.log_path, 'w') as f:
                json.dump(summary, f, indent=4)

            return summary

        except Exception as e:
            print(f"[DiskRecorder-{self.camera_id}] Error al cerrar grabación: {e}")
            return None