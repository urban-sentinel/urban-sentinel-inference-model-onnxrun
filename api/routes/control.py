from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()
class CameraControlCmd(BaseModel):
    camera_id: str
    action: str 

class AddCameraCmd(BaseModel):
    camera_id: str
    camera_type: str  
    path: str         

@router.post("/control/camera")
async def control_camera(cmd: CameraControlCmd, request: Request):
    """
    Controla el flujo interno de una cámara que ya tiene un proceso worker corriendo.
    Acciones: start, stop, enable_inference, disable_inference.
    """
    cid = cmd.camera_id
    action = cmd.action.lower()
    control_queues = request.app.state.control_queues

    if cid not in control_queues:
        raise HTTPException(
            status_code=404, 
            detail=f"Operación no válida: La cámara '{cid}' no tiene un proceso activo."
        )

    queue = control_queues[cid]
    print(f"[API] Comando de ejecución para {cid}: {action}")

    if action == "stop":
        queue.put({"command": "STOP"})
        if hasattr(request.app.state, 'latest_jpeg') and cid in request.app.state.latest_jpeg:
            del request.app.state.latest_jpeg[cid]
            
    elif action == "start":
        queue.put({"command": "START"})
        
    elif action == "disable_inference":
        queue.put({"command": "DISABLE_INFERENCE"})
        
    elif action == "enable_inference":
        queue.put({"command": "ENABLE_INFERENCE"})
        
    else:
        raise HTTPException(status_code=400, detail=f"Acción de ejecución '{action}' no reconocida.")

    return {"status": "ok", "camera_id": cid, "action_processed": action}


@router.post("/control/cameras")
async def add_camera_process(cmd: AddCameraCmd, request: Request):
    """
    Endpoint Dinámico: Solicita al Orquestador levantar un nuevo proceso físico para una cámara.
    """
    orch_queue = request.app.state.orchestrator_queue
    control_queues = request.app.state.control_queues

    if cmd.camera_id in control_queues:
        raise HTTPException(
            status_code=400, 
            detail=f"Conflicto: El ID '{cmd.camera_id}' ya está en uso por otra cámara."
        )

    print(f"[API] Solicitando al Orquestador añadir cámara: {cmd.camera_id}")
    
    orch_queue.put({
        "action": "ADD",
        "id": cmd.camera_id,
        "type": cmd.camera_type,
        "path": cmd.path
    })

    return {
        "status": "request_sent", 
        "camera_id": cmd.camera_id, 
        "detail": "La orden de creación ha sido enviada al Orquestador."
    }


@router.delete("/control/cameras/{camera_id}")
async def remove_camera_process(camera_id: str, request: Request):
    """
    Endpoint Dinámico: Solicita al Orquestador matar y limpiar el proceso de una cámara.
    """
    orch_queue = request.app.state.orchestrator_queue
    control_queues = request.app.state.control_queues

    if camera_id not in control_queues:
        raise HTTPException(
            status_code=404, 
            detail=f"Error al eliminar: No existe un proceso activo para la cámara '{camera_id}'."
        )

    print(f"[API] Solicitando al Orquestador eliminar cámara: {camera_id}")

    orch_queue.put({
        "action": "REMOVE",
        "id": camera_id
    })

    return {
        "status": "request_sent", 
        "camera_id": camera_id, 
        "detail": "La orden de eliminación ha sido enviada al Orquestador."
    }