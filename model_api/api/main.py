# model_api/api/main.py
import asyncio
import sys
import os
import base64
import multiprocessing as mp
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from typing import Dict, Union
from contextlib import asynccontextmanager

# Agregamos la raíz del proyecto ('model_api') al path de Python
model_api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_api_root)

try:
    from api.event_manager import event_manager_task
    from api.connection_manager import ConnectionManager
except ImportError as e:
    print(f"Error fatal en 'main.py': No se pudo importar 'event_manager' o 'connection_manager'. {e}")
    sys.exit(1)

# --- Colas inyectadas por run_app.py ---
inference_queue: Union[mp.Queue, None] = None
results_queue: Union[mp.Queue, None] = None
control_queues: Dict[str, mp.Queue] = {}
video_frames_queue: Union[mp.Queue, None] = None   # << NUEVO

# --- Estado para el WS de video ---
_latest_jpeg: Dict[str, bytes] = {}
_cam_event: Dict[str, asyncio.Event] = defaultdict(asyncio.Event)

# --- App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Servidor FastAPI iniciando...")

    if results_queue is None:
        print("[API] CRÍTICO: Las colas no fueron inyectadas por run_app.py. Saliendo.")
        sys.exit(1)

    # Tarea de eventos (tu WS de inferencia)
    asyncio.create_task(event_manager_task(
        manager=manager,
        results_queue=results_queue,
        control_queues=control_queues
    ))

    # Tarea consumidora de la cola de video (si está inyectada)
    if video_frames_queue is not None:
        async def consume_video_queue():
            loop = asyncio.get_running_loop()
            while True:
                try:
                    # .get() bloqueante fuera del loop
                    item = await loop.run_in_executor(None, video_frames_queue.get)
                    cam = item.get("camera_id")
                    jpg = item.get("jpeg")
                    if cam and jpg:
                        _latest_jpeg[cam] = jpg
                        _cam_event[cam].set()
                except Exception as e:
                    print("[API] Error consumiendo video_frames_queue:", e)
                    await asyncio.sleep(0.05)
        asyncio.create_task(consume_video_queue())
    else:
        print("[API] AVISO: video_frames_queue no está inyectada. El WS de video no enviará frames.")

    yield
    print("[API] Servidor FastAPI apagándose.")

app = FastAPI(
    title="UrbanSentinel API",
    description="API para la detección de violencia en tiempo real.",
    lifespan=lifespan
)

manager = ConnectionManager()

# --- WS de eventos (tu endpoint original) ---
@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, camera_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)

# --- WS de video (frames JPEG en base64) ---
@app.websocket("/ws/frames/{camera_id}")
async def ws_frames(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        # manda el último inmediatamente si existe
        last = _latest_jpeg.get(camera_id)
        if last:
            await websocket.send_json({
                "type": "frame",
                "camera_id": camera_id,
                "jpeg_base64": base64.b64encode(last).decode("ascii"),
            })

        while True:
            # espera a que llegue un nuevo frame para esa cámara
            await _cam_event[camera_id].wait()
            _cam_event[camera_id].clear()

            jpg = _latest_jpeg.get(camera_id)
            if not jpg:
                continue

            await websocket.send_json({
                "type": "frame",
                "camera_id": camera_id,
                "jpeg_base64": base64.b64encode(jpg).decode("ascii"),
            })
    except WebSocketDisconnect:
        pass

@app.get("/")
def read_root():
    return {"message": "UrbanSentinel API en funcionamiento."}
