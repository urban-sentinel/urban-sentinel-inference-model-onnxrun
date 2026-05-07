import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

@router.websocket("/ws/{camera_id}")
async def alerts_websocket(websocket: WebSocket, camera_id: str):
    manager = websocket.app.state.connection_manager
    await manager.connect(websocket, camera_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)

@router.websocket("/ws/frames/{camera_id}")
async def ws_frames(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    
    latest_jpeg = websocket.app.state.latest_jpeg
    cam_event = websocket.app.state.cam_event

    try:
        last = latest_jpeg.get(camera_id)
        if last:
            await websocket.send_json({
                "type": "frame",
                "camera_id": camera_id,
                "jpeg_base64": base64.b64encode(last).decode("ascii"),
            })

        while True:
            await cam_event[camera_id].wait()
            cam_event[camera_id].clear()

            jpg = latest_jpeg.get(camera_id)
            if not jpg:
                continue

            await websocket.send_json({
                "type": "frame",
                "camera_id": camera_id,
                "jpeg_base64": base64.b64encode(jpg).decode("ascii"),
            })
    except WebSocketDisconnect:
        pass