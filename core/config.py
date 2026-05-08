import os

class Config:

    # RUTAS DEL SISTEMA 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SAVE_CLIP_PATH = os.path.join(DATA_DIR, "clips_guardados")
    SAVE_LOG_PATH = os.path.join(DATA_DIR, "logs_eventos")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "ai_engine", "weights")
    
    TEMPORAL_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")
    SPATIAL_MODEL_PATH = os.path.join(WEIGHTS_DIR, "yolo11s.onnx") 

    # MOTOR ESPACIAL (YOLO) 
    YOLO_PERSON_CLASS_ID = 0  
    YOLO_CONFIDENCE = 0.35           
    YOLO_NUM_THREADS = 2            
    TRACKER_MAX_AGE = 30 
    CLUSTER_MARGIN_PX = 50       
    BBOX_SMOOTHING_ALPHA = 0.5   # 0.1 (Muy fluido/Lento) a 0.9 (Muy tembloroso/Rápido)
    
    # MOTOR TEMPORAL (DINOv3) Y TUBELETS 
    CLASSES = ["Golpe", "Peaton", "Patada", "Forcejeo"]
    NUM_FRAMES = 16            
    INPUT_CROP_SIZE = 224
    BBOX_PADDING_PCT = 0.3
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # PIPELINE Y LÓGICA DE EVENTOS 
    TARGET_FPS = 30                 
    STRIDE = 16                     
    PRE_ROLL_SECONDS = 5           
    ALERT_THRESHOLD_CLASSES = [0.400, 0.700, 0.806, 0.500]
    SMOOTHING_WINDOW = 2
    TOLERANCE_FRAMES = 1
    COOLDOWN_SECONDS = 3
    TRACK_PATIENCE_FRAMES = 10

    # OPTIMIZACIÓN DE HARDWARE 
    MAX_BATCH_SIZE = 16
    BATCH_TIMEOUT_SECONDS = 0.1     
    INFERENCE_PROVIDERS = [
        'CUDAExecutionProvider',    
        'DmlExecutionProvider',    
        'CPUExecutionProvider'     
    ]

    

config = Config()

os.makedirs(config.SAVE_CLIP_PATH, exist_ok=True)
os.makedirs(config.SAVE_LOG_PATH, exist_ok=True)