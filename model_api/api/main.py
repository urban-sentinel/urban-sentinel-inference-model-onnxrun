# model_api/api/main.py
import asyncio
import sys
import os
import base64
import multiprocessing as mp
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,          # ok SIEMPRE que no uses "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges"],  # <- el <video> los necesita
)


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

# ... imports existentes ...
from pydantic import BaseModel

# --- MODELO DE DATOS ---
class CameraControlCmd(BaseModel):
    camera_id: str
    action: str # "start" o "stop"

# ... (resto del código setup) ...
# En main.py (Inferencia API)

@app.post("/control/camera")
async def control_camera(cmd: CameraControlCmd):
    """
    Acciones soportadas:
    - "start": Enciende video y todo.
    - "stop": Apaga video y todo (Pantalla negra).
    - "enable_inference": Activa detección IA (Video sigue fluido).
    - "disable_inference": Desactiva detección IA (Video sigue fluido, ahorra GPU).
    """
    cid = cmd.camera_id
    action = cmd.action.lower() # Normalizamos a minúsculas

    if cid not in control_queues:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")

    queue = control_queues[cid]
    
    print(f"[API 8010] Comando recibido para {cid}: {action}")

    # Mapeo de acción externa -> comando interno del worker
    if action == "stop":
        queue.put({"command": "STOP"})
        if cid in _latest_jpeg: del _latest_jpeg[cid] # Limpiar preview
        
    elif action == "start":
        queue.put({"command": "START"})
        
    elif action == "disable_inference":
        queue.put({"command": "DISABLE_INFERENCE"})
        
    elif action == "enable_inference":
        queue.put({"command": "ENABLE_INFERENCE"})
        
    else:
        raise HTTPException(status_code=400, detail=f"Acción '{action}' no válida")

    return {"status": "ok", "camera_id": cid, "action_processed": action}

# --- NUEVO ENDPOINT DE CONTROL DIRECTO ---
@app.post("/control/camera")
async def control_camera(cmd: CameraControlCmd):
    """
    Endpoint para Pausar/Reanudar la cámara.
    El Frontend llama a esto DIRECTAMENTE al puerto 8010.
    """
    cid = cmd.camera_id
    action = cmd.action.upper()

    if cid not in control_queues:
        # Ojo: Si la cámara no existe en run_app.py, no podemos controlarla.
        raise HTTPException(status_code=404, detail="Cámara no encontrada o no inicializada")

    queue = control_queues[cid]

    print(f"[API 8010] Enviando comando {action} a la cola de {cid}")
    
    # Ponemos el mensaje exacto que camera_worker espera
    queue.put({"command": action})

    # Limpieza visual (Opcional): Si paramos, borramos la última imagen para que no se quede 'pegada'
    if action == "STOP" and cid in _latest_jpeg:
        del _latest_jpeg[cid]

    return {"status": "ok", "camera_id": cid, "action_sent": action}

@app.get("/")
def read_root():
    return {"message": "UrbanSentinel API en funcionamiento."}

