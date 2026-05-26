import queue
import time
import os
import sys
import cv2
from multiprocessing import Queue
from queue import Empty
from collections import deque
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config
from io_adapters.readers.webcam_reader import WebcamReader
from io_adapters.readers.file_reader import FileReader
from io_adapters.readers.rtsp_reader import RtspReader

def run_camera_worker(
    camera_id: str,
    reader_type: str, 
    source_path: Any, 
    inference_queue: Queue,
    control_queue: Queue,
    video_frames_queue: Queue, 
    recording_queue: Queue, 
    results_queue: Queue
):
    print(f"[Worker-{camera_id}] Proceso de Ingesta (Tonto) Iniciado.")

    try:
        if reader_type == "file": stream_reader = FileReader(source_path)
        elif reader_type == "webcam": stream_reader = WebcamReader(source_path, 640, 480, config.TARGET_FPS)
        elif reader_type == "rtsp": stream_reader = RtspReader(source_path)
        else: raise ValueError(f"Lector desconocido: {reader_type}")
    except Exception as e:
        print(f"[Worker-{camera_id}] CRÍTICO: Fallo al conectar cámara: {e}")
        return

    source_fps = stream_reader.get_fps() or config.TARGET_FPS
    delay_por_frame = 1.0 / source_fps
    
    is_processing = True  
    is_recording = False  
    is_inferring = True   
    
    pre_roll_buffer = deque(maxlen=int(config.PRE_ROLL_SECONDS * source_fps))
    last_send = 0.0
    frame_counter = 0

    print(f"[Worker-{camera_id}] Corriendo en modo ultra-ligero (Lectura y Compresión).")

    try:
        while True:
            loop_start = time.time()

            while not control_queue.empty():
                try:
                    msg = control_queue.get_nowait()
                    cmd = msg.get("command")
                    
                    if cmd == "START": is_processing = True
                    elif cmd == "STOP": is_processing = False
                    
                    elif cmd == "START_RECORDING" and not is_recording:
                        is_recording = True
                        recording_queue.put(("START", camera_id, list(pre_roll_buffer), source_fps))
                    elif cmd == "STOP_RECORDING" and is_recording:
                        is_recording = False
                        recording_queue.put(("STOP", camera_id))
                        
                    elif cmd == "DISABLE_INFERENCE" and is_inferring:
                        is_inferring = False
                        print(f"[Worker-{camera_id}] Inferencia DESACTIVADA. Liberando GPU...")
                       
                        try: inference_queue.put_nowait({"command": "REMOVE_CAMERA", "camera_id": camera_id})
                        except queue.Full: pass
                        
                    elif cmd == "ENABLE_INFERENCE" and not is_inferring:
                        is_inferring = True
                        print(f"[Worker-{camera_id}] Inferencia ACTIVADA.")
                        
                except Empty: break

            if not is_processing:
                time.sleep(0.1); continue

            ret, frame = stream_reader.read()
            if not ret or frame is None:
                if reader_type == "file": break
                time.sleep(0.1); continue

            frame_counter += 1
            pre_roll_buffer.append(frame)

            if frame_counter % 150 == 0:
                print(f"[Worker-{camera_id}] Heartbeat: Leídos {frame_counter} frames. Stream RTSP estable.")

            h, w = frame.shape[:2]
            
            process_frame = frame
            if w > 1920:
                scale = 1920 / float(w)
                process_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            ok, encoded_frame = cv2.imencode(".jpg", process_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            if ok:
                if is_inferring:
                    try:
                        inference_queue.put_nowait({
                            "camera_id": camera_id,
                            "jpeg_bytes": encoded_frame.tobytes(),
                            "frame_idx": frame_counter
                        })
                    except queue.Full:
                        pass
                
                now = time.time()
                if now - last_send >= 0.033 and video_frames_queue is not None:
                    if w > 640:
                        scale = 640 / float(w)
                        preview_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    else:
                        preview_frame = frame
                    
                    ok_prev, buf_prev = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ok_prev:
                        try: video_frames_queue.put_nowait({"camera_id": camera_id, "jpeg": buf_prev.tobytes()})
                        except queue.Full: pass
                    last_send = now

            if is_recording:
                recording_queue.put(("FRAME", camera_id, frame, {}))

            if reader_type != "file":
                elapsed = time.time() - loop_start
                sleep = delay_por_frame - elapsed
                if sleep > 0: time.sleep(sleep)

    except KeyboardInterrupt: pass
    except Exception as e: print(f"[Worker-{camera_id}] Error fatal: {e}")
    finally:
        print(f"[Worker-{camera_id}] Apagando...")
        if stream_reader: stream_reader.release()
        if is_recording: recording_queue.put(("STOP", camera_id))