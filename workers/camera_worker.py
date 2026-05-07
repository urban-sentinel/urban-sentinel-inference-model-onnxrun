import queue
import time
import os
import sys
import cv2
import numpy as np
from multiprocessing import Queue
from queue import Empty
from collections import deque
from typing import Dict, List, Any, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config
from ai_engine.models.yolo_spatial import YoloSpatialTracker
from ai_engine.pipeline.video_processor import TubeletProcessor
from ai_engine.pipeline.spatial_filter import SpatialFilter
from io_adapters.readers.webcam_reader import WebcamReader
from io_adapters.readers.file_reader import FileReader
from io_adapters.readers.rtsp_reader import RtspReader


def get_interaction_boxes(boxes_dict: Dict[int, List[int]], margin_px: int = 30) -> Dict[str, List[int]]:
    """
    [NUEVA LÓGICA DE CLUSTERING]
    Evalúa todas las personas detectadas. Si sus "zonas de influencia" (caja + margen) 
    se tocan, las fusiona en una única 'Super Caja'.
    Filtro Estricto: Si la persona está sola, la ignora completamente (ahorro de GPU).
    """
    if len(boxes_dict) < 2:
        return {} 

    active_ids = list(boxes_dict.keys())
    merged_groups = []
    visited = set()

    def boxes_are_close(b1, b2, m):
        return not (b1[0]-m > b2[2]+m or b2[0]-m > b1[2]+m or b1[1]-m > b2[3]+m or b2[1]-m > b1[3]+m)

    for i in range(len(active_ids)):
        id1 = active_ids[i]
        if id1 in visited: continue

        current_group = {id1}
        queue = [id1]
        visited.add(id1)

        while queue:
            curr_id = queue.pop(0)
            box1 = boxes_dict[curr_id]

            for j in range(i + 1, len(active_ids)):
                id2 = active_ids[j]
                if id2 not in visited:
                    box2 = boxes_dict[id2]
                    if boxes_are_close(box1, box2, margin_px):
                        visited.add(id2)
                        current_group.add(id2)
                        queue.append(id2)

        if len(current_group) >= 2:
            merged_groups.append(current_group)

    super_boxes = {}
    for group in merged_groups:
   
        group_id = "group_" + "_".join(str(tid) for tid in sorted(group))
        
        min_x = min(boxes_dict[tid][0] for tid in group)
        min_y = min(boxes_dict[tid][1] for tid in group)
        max_x = max(boxes_dict[tid][2] for tid in group)
        max_y = max(boxes_dict[tid][3] for tid in group)
        
        super_boxes[group_id] = [min_x, min_y, max_x, max_y]

    return super_boxes


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
    print(f"[Worker-{camera_id}] Proceso de Ingesta Iniciado.")

    try:
        spatial_tracker = YoloSpatialTracker()
    except Exception as e:
        print(f"[Worker-{camera_id}] CRÍTICO: Fallo al iniciar YOLO: {e}")
        return

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
    
    frame_buffer = deque(maxlen=config.NUM_FRAMES)
    
    track_history: Dict[str, deque] = {} 

    group_patience: Dict[str, int] = {}
    patience_limit = getattr(config, 'TRACK_PATIENCE_FRAMES', 5)
    
    pre_roll_buffer = deque(maxlen=int(config.PRE_ROLL_SECONDS * source_fps))
    
    last_send = 0.0
    frame_counter = 0

    alpha = getattr(config, 'BBOX_SMOOTHING_ALPHA', 0.5)
    yolo_smoother = SpatialFilter(alpha=alpha)
    super_box_smoother = SpatialFilter(alpha=alpha)

    print(f"[Worker-{camera_id}] Corriendo. Ventana temporal: {config.NUM_FRAMES} frames.")

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
                except Empty: break

            if not is_processing:
                time.sleep(0.1); continue

            ret, frame = stream_reader.read()
            if not ret or frame is None:
                if reader_type == "file": break
                time.sleep(0.1); continue

            frame_counter += 1
            frame_buffer.append(frame)
            pre_roll_buffer.append(frame)

            raw_boxes = spatial_tracker.track_persons(frame)
            smoothed_raw_boxes = yolo_smoother.smooth_boxes(raw_boxes)
            margin = getattr(config, 'CLUSTER_MARGIN_PX', 30)

            interaction_boxes = get_interaction_boxes(smoothed_raw_boxes, margin_px=margin)

            smoothed_interaction_boxes = super_box_smoother.smooth_boxes(interaction_boxes)

            active_groups = list(smoothed_interaction_boxes.keys())
            
            for group_id, super_box in smoothed_interaction_boxes.items():
                if group_id not in track_history:
                    track_history[group_id] = deque(maxlen=config.NUM_FRAMES)
                    pad_length = len(frame_buffer) - 1
                    for _ in range(pad_length):
                        track_history[group_id].append(super_box)
                
                track_history[group_id].append(super_box)
                group_patience[group_id] = 0  

            dead_groups = [gid for gid in track_history.keys() if gid not in active_groups]
            for gid in dead_groups:
                group_patience[gid] = group_patience.get(gid, 0) + 1
                
                if group_patience[gid] > patience_limit:
                    del track_history[gid]
                    del group_patience[gid]
                else:
                    last_known_box = track_history[gid][-1]
                    track_history[gid].append(last_known_box)

            if len(frame_buffer) == config.NUM_FRAMES and (frame_counter % config.STRIDE == 0):
                
                valid_tubelets = []
                
                for group_id, box_history in track_history.items():
                    if len(box_history) == config.NUM_FRAMES:
                        
                        tubelet_tensor = TubeletProcessor.create_tubelet(
                            frames=list(frame_buffer),
                            boxes=list(box_history)
                        )
                        
                        valid_tubelets.append({
                            "track_id": group_id,
                            "tensor": tubelet_tensor
                        })

                if valid_tubelets:
                    inference_queue.put((camera_id, valid_tubelets))
                else:
                    results_queue.put((camera_id, {"status": "clear", "data": {}}))

            if is_recording:
                recording_queue.put(("FRAME", camera_id, frame, smoothed_interaction_boxes))

            now = time.time()
            if now - last_send >= 0.10 and video_frames_queue is not None:
                preview_frame = frame.copy()
                
                for tid, b in raw_boxes.items():
                    cv2.rectangle(preview_frame, (b[0], b[1]), (b[2], b[3]), (255, 100, 0), 1)
                
                for gid, b in smoothed_interaction_boxes.items():
                    cv2.rectangle(preview_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
                    cv2.putText(preview_frame, f"ZONA: {gid}", (b[0], max(0, b[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                h, w = preview_frame.shape[:2]
                if w > 640:
                    scale = 640 / float(w)
                    preview_frame = cv2.resize(preview_frame, (int(w*scale), int(h*scale)))
                
                ok, buf = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ok:
                    try: video_frames_queue.put_nowait({"camera_id": camera_id, "jpeg": buf.tobytes()})
                    except queue.Full: pass
                last_send = now

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