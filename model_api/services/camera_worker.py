import time
import os
import sys
from multiprocessing import Queue
from queue import Empty
from collections import deque
import numpy as np
from typing import Union, List, Any

import cv2
import queue  # para queue.Full

# Agregamos la raíz del proyecto ('model_api') al path de Python
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from config import config
    from processing.video_processor import preprocess_clip
    from services.event_recorder import EventRecorder
    from services.stream_reader.file_reader import FileReader
    from services.stream_reader.base_reader import BaseReader
    from services.stream_reader.webcam_reader import WebcamReader
    
    # --- [NUEVO] Importamos el wrapper del detector de personas ---
    from onnx_model.onnx_person_detector import PersonDetector

except ImportError as e:
    print(f"Error fatal en 'camera_worker.py': No se pudo importar un módulo. {e}")
    sys.exit(1)

def run_camera_worker(
    camera_id: str,
    reader_type: str, 
    source_path: Any, 
    inference_queue: Queue,
    control_queue: Queue,
    video_frames_queue: Queue, 
    results_queue: Queue
):
    print(f"[Worker-{camera_id}] Proceso iniciado.")

    stream_reader: Union[BaseReader, None] = None 
    current_recorder: Union[EventRecorder, None] = None
    person_detector: Union[PersonDetector, None] = None  # --- [NUEVO] Variable para el detector
    
    # Parámetros visuales
    last_send = 0.0
    SEND_INTERVAL = 0.10
    MAX_W = 640            
    JPEG_QUALITY = 75      
    
    # --- BANDERAS DE ESTADO ---
    is_processing = True      # Interruptor Maestro: ¿Leemos cámara?
    inference_enabled = True  # Interruptor IA: ¿Mandamos a inferencia?

    try:
        # --- [NUEVO] 1. Cargar el detector de personas (CPU) ---
        try:
            person_detector = PersonDetector()
            print(f"[Worker-{camera_id}] Detector de personas (YOLOv8n-CPU) inicializado.")
        except Exception as e:
            print(f"[Worker-{camera_id}] CRÍTICO: No se pudo cargar PersonDetector: {e}")
            # Nota: Dependiendo de tu lógica, podrías querer hacer 'return' aquí si es obligatorio.
            # Por ahora dejamos que continúe, pero el filtro fallará.

        # --- 2. Inicialización de Reader y Buffers ---
        print(f"[Worker-{camera_id}] Iniciando lector tipo '{reader_type}'")
        
        if reader_type == "file":
            stream_reader = FileReader(source_path)
        elif reader_type == "webcam":
            # Ajusta índice o path según corresponda
            stream_reader = WebcamReader(source_path, 640, 480, 30)
        else:
            # Fallback o raise
            raise ValueError(f"Tipo desconocido: {reader_type}")

        source_fps = stream_reader.get_fps()
        if not source_fps:
            source_fps = config.TARGET_FPS
        
        CLIP_DURATION_SEC = config.CLIP_LEN / config.TARGET_FPS
        INFERENCE_BUFFER_SIZE = int(CLIP_DURATION_SEC * source_fps)
        PRE_ROLL_BUFFER_SIZE = int(config.PRE_ROLL_SECONDS * source_fps)

        inference_buffer = deque(maxlen=INFERENCE_BUFFER_SIZE)
        pre_roll_buffer = deque(maxlen=PRE_ROLL_BUFFER_SIZE)
        frame_counter = 0
        delay_por_frame = 1.0 / source_fps
        last_known_probs = np.array([0.0] * len(config.CLASSES))

        # --- 3. Bucle Principal ---
        while True:
            # A. REVISAR COLA DE CONTROL
            while not control_queue.empty():
                try:
                    msg = control_queue.get_nowait()
                    
                    if isinstance(msg, dict):
                        cmd = msg.get("command")
                        # --- COMANDOS DE STREAM (Video) ---
                        if cmd == "STOP":
                            is_processing = False
                            if current_recorder: current_recorder.close(); current_recorder = None
                            print(f"[Worker-{camera_id}] ⏸️ STREAM PAUSADO.")
                        elif cmd == "START":
                            is_processing = True
                            print(f"[Worker-{camera_id}] ▶️ STREAM REANUDADO.")
                        
                        # --- COMANDOS DE INFERENCIA (IA) ---
                        elif cmd == "DISABLE_INFERENCE":
                            inference_enabled = False
                            print(f"[Worker-{camera_id}] 🧠 IA DESACTIVADA (Solo video).")
                        elif cmd == "ENABLE_INFERENCE":
                            inference_enabled = True
                            print(f"[Worker-{camera_id}] 🧠 IA ACTIVADA (Detectando).")

                    elif isinstance(msg, str):
                        if msg == "START_RECORDING" and is_processing:
                            if not current_recorder:
                                current_recorder = EventRecorder(camera_id, list(pre_roll_buffer), source_fps)
                                current_recorder.start()
                        elif msg == "STOP_RECORDING" and current_recorder:
                            res = current_recorder.close(); current_recorder = None
                            if res: results_queue.put({"type": "event_complete", **res})
                    
                    elif isinstance(msg, np.ndarray):
                        last_known_probs = msg

                except Empty: break

            # B. SI EL STREAM ESTÁ PAUSADO, DORMIR
            if not is_processing:
                time.sleep(0.1)
                continue

            # --- PROCESAMIENTO ---
            loop_start_time = time.time()
            
            ret, frame = stream_reader.read()
            if not ret:
                print(f"[Worker-{camera_id}] Fallo lectura frame o fin de archivo.")
                # Dependiendo de si es loop infinito o no:
                if reader_type == "file": break
                time.sleep(0.5); continue
            
            frame_counter += 1
            inference_buffer.append(frame)
            pre_roll_buffer.append(frame)

            # 1. Preview (SIEMPRE corre si is_processing=True)
            now = time.time()
            if now - last_send >= SEND_INTERVAL and video_frames_queue is not None:
                h, w = frame.shape[:2]
                if w > MAX_W:
                    scale = MAX_W / float(w)
                    frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                else: frame_small = frame
                
                ok, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok:
                    try: video_frames_queue.put_nowait({"camera_id": camera_id, "jpeg": buf.tobytes()})
                    except queue.Full: pass
                last_send = now

            # 2. Grabador (Corre si hay evento activo)
            if current_recorder: current_recorder.add_frame(frame, last_known_probs)

            # 3. Inferencia (AHORA CON FILTRO DE PERSONAS)
            if inference_enabled:
                if (len(inference_buffer) == INFERENCE_BUFFER_SIZE and frame_counter % config.STRIDE == 0):
                    
                    # --- [NUEVO] Lógica de filtrado YOLO ---
                    person_count = -1
                    try:
                        if person_detector:
                            person_count = person_detector.count_persons(frame)
                        else:
                            # Si falló la carga, asumimos -1 para no bloquear o >=2 para forzar
                            pass 
                    except Exception as e:
                        print(f"[Worker-{camera_id}] Warn YOLO: {e}")

                    # DECISIÓN:
                    if person_count >= 2:
                        # --- CASO A: Hay gente -> ENVIAR A GPU ---
                        try:
                            tensor = preprocess_clip(list(inference_buffer))
                            if np.isfinite(tensor).all():
                                inference_queue.put((camera_id, tensor))
                        except Exception as e:
                            print(f"[Worker-{camera_id}] Err Inf: {e}")

                    elif person_count < 0:
                        # --- CASO B: Error en YOLO -> No hacemos nada (logueado arriba) ---
                        pass

                    else:
                        # --- CASO C: < 2 Personas -> BYPASS (Optimización) ---
                        # Enviamos resultado neutral directo al manager para mantener vivo el status
                        neutral_probs = np.array([0.0] * len(config.CLASSES))
                        
                        # IMPORTANTE: Actualizamos last_known_probs para que si se graba algo manual, 
                        # no tenga probabilidades "viejas" de una pelea anterior.
                        last_known_probs = neutral_probs
                        
                        # Enviamos a la cola de resultados (saltándonos la GPU)
                        try:
                            results_queue.put((camera_id, neutral_probs))
                        except Exception:
                            pass # Cola llena o error, no crítico
            
            # Control FPS
            elapsed = time.time() - loop_start_time
            sleep = delay_por_frame - elapsed
            if sleep > 0: time.sleep(sleep)

    except (KeyboardInterrupt, SystemExit): pass
    except Exception as e:
        print(f"[Worker-{camera_id}] CRÍTICO NO CONTROLADO: {e}")
    finally:
        print(f"[Worker-{camera_id}] Cerrando recursos...")
        if current_recorder: current_recorder.close()
        if stream_reader: stream_reader.release()