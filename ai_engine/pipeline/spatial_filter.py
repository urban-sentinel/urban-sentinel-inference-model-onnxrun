from typing import Dict, List, Union

class SpatialFilter:
    """
    Filtro espacial (Exponential Moving Average) para suavizar las coordenadas 
    de las Bounding Boxes en el tiempo y eliminar el 'Jittering' (parpadeo).
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.history: Dict[Union[int, str], List[int]] = {}

    def smooth_boxes(self, current_boxes: Dict[Union[int, str], List[int]]) -> Dict[Union[int, str], List[int]]:
        """
        Aplica el suavizado a un diccionario de cajas.
        Soporta IDs numéricos (YOLO) y strings (Super Cajas).
        """
        smoothed_boxes = {}
        
        for box_id, coords in current_boxes.items():
            if box_id not in self.history:
                self.history[box_id] = coords
            else:
                prev_coords = self.history[box_id]
                new_coords = [
                    int(self.alpha * curr + (1 - self.alpha) * prev)
                    for curr, prev in zip(coords, prev_coords)
                ]
                self.history[box_id] = new_coords
                
            smoothed_boxes[box_id] = self.history[box_id]

        dead_keys = [k for k in self.history.keys() if k not in current_boxes]
        for k in dead_keys:
            del self.history[k]

        return smoothed_boxes