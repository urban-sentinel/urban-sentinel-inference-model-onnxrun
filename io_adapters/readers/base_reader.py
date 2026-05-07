from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Optional

class BaseReader(ABC):
    """
    Contrato abstracto para todos los lectores de video.
    """
    
    @abstractmethod
    def __init__(self, source: Any):
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retorna (True, frame) si es exitoso.
        Retorna (False, None) si falló o terminó.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        pass

    @abstractmethod
    def release(self) -> None:
        pass