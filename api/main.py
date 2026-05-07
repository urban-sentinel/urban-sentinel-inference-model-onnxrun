import asyncio
import sys
import os
import queue
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routes import websocket, control
from api.services.connection_manager import ConnectionManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Servidor FastAPI iniciando (V2 Espacio-Temporal)...")

    app.state.latest_jpeg = {}
    app.state.cam_event = defaultdict(asyncio.Event)
    app.state.connection_manager = ConnectionManager()

    if not hasattr(app.state, "results_queue") or not hasattr(app.state, "video_frames_queue"):
        print("[API] CRÍTICO: Las colas no fueron inyectadas en app.state por run_app.py.")
    else:
        from api.services.event_manager import event_manager_task
        
        asyncio.create_task(event_manager_task(
            manager=app.state.connection_manager,
            results_queue=app.state.results_queue,
            control_queues=app.state.control_queues
        ))

        async def consume_video_queue():
            loop = asyncio.get_running_loop()
            while True:
                try:
                   
                    item = await loop.run_in_executor(None, lambda: app.state.video_frames_queue.get(timeout=0.2))
                    cam = item.get("camera_id")
                    jpg = item.get("jpeg")
                    if cam and jpg:
                        app.state.latest_jpeg[cam] = jpg
                        app.state.cam_event[cam].set()
                        
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if "Event loop is closed" not in str(e):
                        print(f"[API] Error consumiendo video_frames_queue: {e}")
                    await asyncio.sleep(0.05)
        
        asyncio.create_task(consume_video_queue())

    yield
    print("[API] Servidor FastAPI apagándose.")

def create_app() -> FastAPI:
    app = FastAPI(
        title="UrbanSentinel API",
        description="API V2 Espacio-Temporal",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges"],
    )

    app.include_router(websocket.router)
    app.include_router(control.router)

    @app.get("/")
    def read_root():
        return {"message": "UrbanSentinel API V2 en funcionamiento."}

    return app

app = create_app()