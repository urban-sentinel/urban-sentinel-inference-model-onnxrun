import os
import sys
import cv2
import numpy as np
import torch
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config

class TubeletProcessor:
    """
    Motor matemático de procesamiento de video (CPU).
    Aísla a una persona en una secuencia de frames y genera el tensor para DINOv3.
    """

    @staticmethod
    def create_tubelet(frames: List[np.ndarray], boxes: List[List[int]]) -> torch.Tensor:
        """
        Transforma una secuencia de imágenes y cajas en un Tensor de PyTorch normalizado.
        
        Args:
            frames: Lista de 16 matrices Numpy (los frames crudos BGR).
            boxes: Lista de 16 cajas [x1, y1, x2, y2] correspondientes a la misma persona.
            
        Returns:
            torch.Tensor: Forma (Channels, Frames, Height, Width) -> (3, 16, 224, 224)
        """
        
        boxes_np = np.array(boxes)
        anchos = boxes_np[:, 2] - boxes_np[:, 0]
        altos = boxes_np[:, 3] - boxes_np[:, 1]
        
        w_max, h_max = np.max(anchos), np.max(altos)
        
        lado_cuadrado = int(max(w_max, h_max) * (1.0 + config.BBOX_PADDING_PCT))
        mitad_lado = lado_cuadrado // 2
        
        recortes_procesados = []
        
        for frame, box in zip(frames, boxes):
            alto_img, ancho_img = frame.shape[:2]
            
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            
            crop_x1 = cx - mitad_lado
            crop_y1 = cy - mitad_lado
            crop_x2 = cx + mitad_lado
            crop_y2 = cy + mitad_lado
            
            valid_x1, valid_y1 = max(0, crop_x1), max(0, crop_y1)
            valid_x2, valid_y2 = min(ancho_img, crop_x2), min(alto_img, crop_y2)
            
            recorte_real = frame[valid_y1:valid_y2, valid_x1:valid_x2]
            
            canvas = np.zeros((lado_cuadrado, lado_cuadrado, 3), dtype=np.uint8)
            paste_x1 = valid_x1 - crop_x1
            paste_y1 = valid_y1 - crop_y1
            
            if recorte_real.size > 0:
                canvas[paste_y1:paste_y1 + recorte_real.shape[0], paste_x1:paste_x1 + recorte_real.shape[1]] = recorte_real
            
            canvas_resized = cv2.resize(canvas, (config.INPUT_CROP_SIZE, config.INPUT_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            canvas_rgb = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2RGB)
            
            recortes_procesados.append(canvas_rgb)
        
        video_np = np.array(recortes_procesados, dtype=np.float32) / 255.0
        
        video_np = (video_np - config.NORM_MEAN) / config.NORM_STD
        
        video_np = np.transpose(video_np, (3, 0, 1, 2))
        
        return torch.from_numpy(video_np).float()

    @staticmethod
    def batch_tubelets(tubelets: List[torch.Tensor]) -> torch.Tensor:
        """
        Agrupa múltiples Tubelets individuales en un solo Batch para acelerar la GPU.
        Ej: 3 personas peleando -> 1 tensor de forma (3, 3, 16, 224, 224)
        """
        return torch.stack(tubelets, dim=0)