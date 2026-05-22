import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Tuple, Any

from core.config import config
from ai_engine.pipeline.spatial_filter import SpatialFilter

class TrackBufferManager:
    """
    Administrador de Contexto Espacio-Temporal.
    Mantiene en RAM el historial de recortes.
    Incluye lógica de centrado y normalización rescatada de video_processor.
    """
    
    def __init__(self):
        self.track_history: Dict[str, Dict[str, deque]] = {}      
        self.group_patience: Dict[str, Dict[str, int]] = {}       
        self.yolo_smoothers: Dict[str, SpatialFilter] = {}
        self.super_box_smoothers: Dict[str, SpatialFilter] = {}
        
        self.patience_limit = getattr(config, 'TRACK_PATIENCE_FRAMES', 10)
        self.margin_px = getattr(config, 'CLUSTER_MARGIN_PX', 30)
        self.alpha = getattr(config, 'BBOX_SMOOTHING_ALPHA', 0.5)

    def _initialize_camera_state(self, camera_id: str):
        if camera_id not in self.track_history:
            self.track_history[camera_id] = {}
            self.group_patience[camera_id] = {}
            self.yolo_smoothers[camera_id] = SpatialFilter(alpha=self.alpha)
            self.super_box_smoothers[camera_id] = SpatialFilter(alpha=self.alpha)

    def remove_camera(self, camera_id: str):
        self.track_history.pop(camera_id, None)
        self.group_patience.pop(camera_id, None)
        self.yolo_smoothers.pop(camera_id, None)
        self.super_box_smoothers.pop(camera_id, None)

    def _get_interaction_boxes(self, boxes_dict: Dict[int, List[int]], margin_px: int) -> Dict[str, List[int]]:
        """Lógica de Clustering por colisión de áreas."""
        if len(boxes_dict) < 2: return {} 

        active_ids = list(boxes_dict.keys())
        merged_groups = []
        visited = set()

        def boxes_are_close(b1, b2, m):
            return not (b1[0]-m > b2[2]+m or b2[0]-m > b1[2]+m or b1[1]-m > b2[3]+m or b2[1]-m > b1[3]+m)

        for i in range(len(active_ids)):
            id1 = active_ids[i]
            if id1 in visited: continue
            current_group = {id1}
            queue_ids = [id1]
            visited.add(id1)

            while queue_ids:
                curr_id = queue_ids.pop(0)
                box1 = boxes_dict[curr_id]
                for j in range(i + 1, len(active_ids)):
                    id2 = active_ids[j]
                    if id2 not in visited:
                        box2 = boxes_dict[id2]
                        if boxes_are_close(box1, box2, margin_px):
                            visited.add(id2)
                            current_group.add(id2)
                            queue_ids.append(id2)

            if len(current_group) >= 2: merged_groups.append(current_group)

        super_boxes = {}
        for group in merged_groups:
            group_id = "group_" + "_".join(str(tid) for tid in sorted(group))
            super_boxes[group_id] = [
                min(boxes_dict[t][0] for t in group), min(boxes_dict[t][1] for t in group),
                max(boxes_dict[t][2] for t in group), max(boxes_dict[t][3] for t in group)
            ]
        return super_boxes

    def _crop_and_resize(self, frame: np.ndarray, box: List[int]) -> np.ndarray:
        """
        [RESTAURADO DEL VIDEO_PROCESSOR ORIGINAL]
        Lógica de Lienzo Cuadrado (Canvas) para evitar deformación de proporciones.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = box
        
        w_box, h_box = (x2 - x1), (y2 - y1)
        lado_max = max(w_box, h_box)
        lado_cuadrado = int(lado_max * (1.0 + config.BBOX_PADDING_PCT))
        mitad_lado = lado_cuadrado // 2
        
        cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
        crop_x1, crop_y1 = cx - mitad_lado, cy - mitad_lado
        crop_x2, crop_y2 = cx + mitad_lado, cy + mitad_lado
        
        valid_x1, valid_y1 = max(0, crop_x1), max(0, crop_y1)
        valid_x2, valid_y2 = min(w_frame, crop_x2), min(h_frame, crop_y2)
        
        recorte_real = frame[valid_y1:valid_y2, valid_x1:valid_x2]
        
        canvas = np.zeros((lado_cuadrado, lado_cuadrado, 3), dtype=np.uint8)
        paste_x1, paste_y1 = valid_x1 - crop_x1, valid_y1 - crop_y1
        
        if recorte_real.size > 0:
            canvas[paste_y1:paste_y1 + recorte_real.shape[0], paste_x1:paste_x1 + recorte_real.shape[1]] = recorte_real
        
        canvas_resized = cv2.resize(canvas, (config.INPUT_CROP_SIZE, config.INPUT_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        canvas_rgb = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2RGB)
        
        return canvas_rgb

    def process_frame(self, camera_id: str, frame: np.ndarray, raw_boxes: Dict[int, List[int]], frame_idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        self._initialize_camera_state(camera_id)
        
        smoothed_raw = self.yolo_smoothers[camera_id].smooth_boxes(raw_boxes)
        interaction_boxes = self._get_interaction_boxes(smoothed_raw, self.margin_px)
        final_boxes = self.super_box_smoothers[camera_id].smooth_boxes(interaction_boxes)
        
        active_groups = list(final_boxes.keys())
        history = self.track_history[camera_id]
        patience = self.group_patience[camera_id]

        for group_id, super_box in final_boxes.items():
            crop = self._crop_and_resize(frame, super_box)
            if group_id not in history:
                history[group_id] = deque([crop] * (config.NUM_FRAMES - 1), maxlen=config.NUM_FRAMES)
            history[group_id].append(crop)
            patience[group_id] = 0

        dead_groups = [gid for gid in history.keys() if gid not in active_groups]
        for gid in dead_groups:
            patience[gid] += 1
            if patience[gid] > self.patience_limit:
                del history[gid]; del patience[gid]
            elif len(history[gid]) > 0:
                history[gid].append(history[gid][-1])

        ready_tubelets = []
        if frame_idx % config.STRIDE == 0:
            for group_id, box_history in history.items():
                if len(box_history) == config.NUM_FRAMES:
                    
                    video_np = np.stack(list(box_history))
                    
                    # --- [RESTAURADO] NORMALIZACIÓN MATEMÁTICA ---
                    video_np = video_np.astype(np.float32) / 255.0
                    mean = np.array(config.NORM_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
                    std = np.array(config.NORM_STD, dtype=np.float32).reshape(1, 1, 1, 3)
                    video_np = (video_np - mean) / std
                    
                    ready_tubelets.append({
                        "track_id": group_id,
                        "tubelet_data": video_np
                    })

        return ready_tubelets, final_boxes