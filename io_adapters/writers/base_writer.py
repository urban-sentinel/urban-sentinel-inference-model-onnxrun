from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional

class BaseWriter(ABC):
    """
    Contrato abstracto para cualquier mecanismo de guardado de video y metadatos.
    """

    @abstractmethod
    def __init__(self, camera_id: str, source_fps: float, output_dir: str):
        pass

    @abstractmethod
    def start_recording(self, pre_roll_frames: List[np.ndarray]) -> bool:
        """
        Inicia la sesión de grabación y guarda los frames de contexto previo.
        Retorna True si se inició correctamente.
        """
        pass

    @abstractmethod
    def write_frame(self, frame: np.ndarray, metadata: Dict[str, float]) -> None:
        """
        Escribe un único frame junto con sus metadatos.
        """
        pass

    @abstractmethod
    def close(self) -> Optional[Dict]:
        """
        Cierra el archivo de video y devuelve un diccionario resumen de la grabación.
        """
        pass