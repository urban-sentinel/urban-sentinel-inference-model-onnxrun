import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from .base_writer import BaseWriter

class DiskRecorder(BaseWriter):
    """
    Escritor sincrónico rápido. Escribe el frame directamente al disco usando cv2.VideoWriter.
    Está diseñado para ser ejecutado dentro de un Proceso Aislado (Worker de Grabación).
    """

    def __init__(self, camera_id: str, source_fps: float, video_dir: str, log_dir: str):
        self.camera_id = camera_id
        self.source_fps = float(source_fps)
        self.video_dir = video_dir
        self.log_dir = log_dir
        
        self.is_open = False
        self.video_writer: Optional[cv2.VideoWriter] = None
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
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.source_fps, (w, h))
            
            if not self.video_writer.isOpened():
                raise IOError("No se pudo abrir VideoWriter de OpenCV.")

            self.is_open = True
            self.start_time = time.time()
            self.logs = []
            
            print(f"[DiskRecorder-{self.camera_id}] Grabación iniciada: {file_basename}.mp4")
            
            for frame in pre_roll_frames:
                self.video_writer.write(frame)
                
            return True

        except Exception as e:
            print(f"[DiskRecorder-{self.camera_id}] ERROR Crítico al iniciar: {e}")
            self.is_open = False
            return False

    def write_frame(self, frame: np.ndarray, metadata: Dict[str, float]) -> None:
        """
        Escribe el frame actual y almacena los metadatos.
        """
        if not self.is_open or self.video_writer is None:
            return

        try:
            self.video_writer.write(frame)
            
            log_entry = {
                "timestamp_ms": int((time.time() - self.start_time) * 1000),
                "metadata": metadata 
            }
            self.logs.append(log_entry)
            
        except Exception as e:
            print(f"[DiskRecorder-{self.camera_id}] Error al escribir frame: {e}")

    def close(self) -> Optional[Dict]:
        """
        Cierra los recursos, guarda el JSON y retorna el resumen.
        """
        if not self.is_open:
            return None
            
        print(f"[DiskRecorder-{self.camera_id}] Cerrando archivo: {os.path.basename(self.video_path)}...")
        
        try:
            self.is_open = False
            if self.video_writer:
                self.video_writer.release()
            
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