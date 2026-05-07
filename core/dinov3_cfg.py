import os

class ModelConfig:

    # Rutas del Sistema 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
    OUTPUT_DIR = os.path.join(BASE_DIR, "core", "checkpoints")
    VAL_LIST_FILE = os.path.join(OUTPUT_DIR, "lista_validacion.txt")

    # Configuración del Dataset 
    VAL_SPLIT = 0.20
    RANDOM_SEED = 42

    # Orden estricto PyTorch (0: Golpe, 1: No Violencia, 2: Patada, 3: Forcejeo)
    CLASSES = ["punch", "pedestrian", "kick", "struggle"]

    # Arquitectura del Modelo 
    MODEL_NAME = "vit_small_patch16_dinov3" 
    NUM_CLASSES = 4                 
    PRETRAINED = True               
    DROPOUT_RATE = 0.3

    # Extracción Espacio-Temporal 
    NUM_FRAMES = 16                 
    MIN_FRAMES_PER_TUBELET = 10     
    IMAGE_SIZE = 224                
    PADDING_TUBELET = 0.20          

    # Hardware y Optimización (I/O y GPU)
    EPOCHS = 100  
    EARLY_STOPPING_PATIENCE = 15
    BATCH_SIZE = 8              
    ACCUMULATION_STEPS = 8         
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    PIN_MEMORY = True
    USE_AMP = True
    CUDNN_BENCHMARK = True                 

    # Hiperparámetros Base 
    LEARNING_RATE = 2.522384725797283e-05       
    WEIGHT_DECAY = 0.00043183111895020145
    FOCAL_GAMMA = 2.0
    CONFIDENCE_THRESHOLD = 0.35
    LABEL_SMOOTHING = 0.2

    CLASS_WEIGHTS = [1.3665, 1.0, 2.1125, 1.0146]

    