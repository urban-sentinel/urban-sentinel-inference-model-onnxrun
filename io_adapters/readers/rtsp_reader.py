import av
import threading
import time
import numpy as np
from typing import Tuple, Optional
from .base_reader import BaseReader

class RtspReader(BaseReader):
    """
    Lector optimizado con PyAV para flujos de red (RTSP).
    Maneja timeouts, protocolo TCP y reconexión automática en hilo dedicado.
    """

    def __init__(self, source: str):
        self.rtsp_url = source
        self.fps = 30.0 
        
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.running = True
        
        self.container = None
        self.video_stream = None
        self.frame_iter = None

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[RtspReader] Iniciando conexión RTSP en segundo plano: {source}")

    def _connect(self):
        """Intenta establecer y configurar la conexión de red de forma segura."""
        if self.container:
            try:
                self.container.close()
            except:
                pass
                
        try:
            self.container = av.open(
                self.rtsp_url, 
                options={'rtsp_transport': 'tcp', 'stimeout': '5000000'}
            )
            self.video_stream = self.container.streams.video[0]
            self.video_stream.thread_type = "AUTO"
            
            if self.video_stream.average_rate:
                self.fps = float(self.video_stream.average_rate)
                
            self.frame_iter = self.container.decode(self.video_stream)
            self.is_connected = True
            print(f"[RtspReader] Conectado exitosamente. FPS: {self.fps}")
            
        except Exception as e:
            print(f"[RtspReader] Error de conexión ({e}). Reintentando en 3s...")
            self.is_connected = False
            time.sleep(3)

    def _update_loop(self):
        """Bucle en segundo plano que siempre mantiene el frame más reciente."""
        while self.running:
            if not self.is_connected:
                self._connect()
                continue

            try:
                frame = next(self.frame_iter)
                img_array = frame.to_ndarray(format='bgr24')

                img_array = np.ascontiguousarray(img_array)
                
                with self.lock:
                    self.last_frame = img_array
                    
            except (StopIteration, av.EOFError, av.error.AVError) as e:
                print(f"[RtspReader] Stream de red interrumpido ({e}). Reconectando...")
                self.is_connected = False
            except Exception as e:
                print(f"[RtspReader] Error inesperado en stream: {e}")
                self.is_connected = False
                time.sleep(1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.last_frame is not None:
                frame_a_enviar = self.last_frame.copy()
                
                self.last_frame = None
                
                return True, frame_a_enviar
            return False, None

    def get_fps(self) -> float:
        return self.fps

    def release(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.container:
            self.container.close()