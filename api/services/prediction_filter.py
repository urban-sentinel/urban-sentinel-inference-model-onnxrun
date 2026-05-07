import collections
from typing import Dict, Tuple
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config

class PredictionFilter:
    def __init__(self):
        self.history: Dict[str, Dict[str, collections.deque]] = {}
        self.tolerance_counter: Dict[str, Dict[str, int]] = {}

    def update_and_check(self, camera_id: str, group_id: str, max_violent_prob: float) -> Tuple[bool, float]:
        """
        Añade la probabilidad actual al historial, calcula la media móvil y verifica la tolerancia.
        Retorna (is_violent_smoothed, smoothed_probability)
        """
        if camera_id not in self.history:
            self.history[camera_id] = {}
            self.tolerance_counter[camera_id] = {}
            
        if group_id not in self.history[camera_id]:
            window = getattr(config, 'SMOOTHING_WINDOW', 5)
            self.history[camera_id][group_id] = collections.deque(maxlen=window)
            self.tolerance_counter[camera_id][group_id] = 0

        self.history[camera_id][group_id].append(max_violent_prob)
        
        current_history = self.history[camera_id][group_id]
        moving_average = sum(current_history) / len(current_history)
        
        threshold = getattr(config, 'ALERT_THRESHOLD', 0.85)
        tolerance_frames = getattr(config, 'TOLERANCE_FRAMES', 2)
        
        if moving_average >= threshold:
            self.tolerance_counter[camera_id][group_id] += 1
        else:
            self.tolerance_counter[camera_id][group_id] = 0
            
        is_violent = self.tolerance_counter[camera_id][group_id] >= tolerance_frames
        
        return is_violent, round(moving_average, 4)
        
    def clear_camera(self, camera_id: str):
        """Limpia la memoria temporal de una cámara cuando la calle se vacía."""
        if camera_id in self.history:
            del self.history[camera_id]
        if camera_id in self.tolerance_counter:
            del self.tolerance_counter[camera_id]