import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config
from ai_engine.models.template.dinov3_model import UrbanSentinelModel

class DinoV3TemporalPredictor:
    """
    Motor Temporal basado en PyTorch (DINOv3).
    Responsable de analizar secuencias de recortes (Tubelets) en la GPU 
    para detectar acciones violentas.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        self._initialize_model()

    def _initialize_model(self):
        """
        Instancia la arquitectura UrbanSentinelModel, carga el state_dict del checkpoint 
        en la VRAM de la tarjeta de video y lo optimiza.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[DINOv3] CRÍTICO: Pesos no encontrados en {self.model_path}")

        try:
            print(f"[DINOv3] Cargando modelo en dispositivo: {self.device}...")
            
            self.model = UrbanSentinelModel().to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.eval()
            
            if hasattr(torch, 'compile') and sys.platform != "win32":
                try:
                    self.model = torch.compile(self.model)
                    print("[DINOv3] Optimización 'torch.compile' aplicada con éxito.")
                except Exception as e:
                    print(f"[DINOv3] Warn: No se pudo compilar el modelo (ignorando): {e}")

            print("[DINOv3] Motor Temporal listo para inferencia.")

        except Exception as e:
            raise RuntimeError(f"[DINOv3] Error fatal al cargar el modelo: {e}")

    @torch.no_grad()
    def predict_batch(self, tubelets_tensor: torch.Tensor) -> np.ndarray:
        """
        Recibe un tensor con un batch de Tubelets y devuelve las probabilidades.
        
        Args:
            tubelets_tensor: Tensor de forma (Batch_Size, Channels, Frames, Height, Width)
                             Ej: (3, 3, 32, 224, 224) -> 3 personas, 32 frames.
                             
        Returns:
            np.ndarray: Matriz de probabilidades de forma (Batch_Size, Num_Classes)
        """
        if self.model is None:
            raise RuntimeError("El modelo no está cargado.")

        try:
            tubelets_tensor = tubelets_tensor.to(self.device, dtype=torch.float32, non_blocking=True)

            logits = self.model(tubelets_tensor)
                
            probabilities = torch.sigmoid(logits)

            return probabilities.cpu().numpy()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[DINOv3] ERROR: ¡Out Of Memory en la GPU! Liberando caché...")
                torch.cuda.empty_cache()
            print(f"[DINOv3] Error de inferencia: {e}")
            return np.zeros((tubelets_tensor.size(0), len(config.CLASSES)), dtype=np.float32)