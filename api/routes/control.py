from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()

class CameraControlCmd(BaseModel):
    camera_id: str
    action: str 

@router.post("/control/camera")
async def control_camera(cmd: CameraControlCmd, request: Request):
    cid = cmd.camera_id
    action = cmd.action.lower()

    control_queues = request.app.state.control_queues

    if cid not in control_queues:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")

    queue = control_queues[cid]
    print(f"[API] Comando recibido para {cid}: {action}")

    if action == "stop":
        queue.put({"command": "STOP"})
        if cid in request.app.state.latest_jpeg: 
            del request.app.state.latest_jpeg[cid] 
            
    elif action == "start":
        queue.put({"command": "START"})
        
    elif action == "disable_inference":
        queue.put({"command": "DISABLE_INFERENCE"})
        
    elif action == "enable_inference":
        queue.put({"command": "ENABLE_INFERENCE"})
        
    else:
        raise HTTPException(status_code=400, detail=f"Acción '{action}' no válida")

    return {"status": "ok", "camera_id": cid, "action_processed": action}