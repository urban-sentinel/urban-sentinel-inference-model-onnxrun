import asyncio
from fastapi import WebSocket
from typing import Dict, List

class ConnectionManager:
    """
    [Capa de Red WebSocket]
    Gestiona todas las conexiones activas entre el Frontend (React/Dashboard) y el Backend.
    Mapea un 'camera_id' específico a una lista de WebSockets suscritos, permitiendo
    enviar alertas solo a los usuarios que están viendo esa cámara.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        """
        Acepta la conexión HTTP y la actualiza a WebSocket.
        Registra al cliente en la "sala" correspondiente a su cámara.
        """
        await websocket.accept()
        
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
            
        self.active_connections[camera_id].append(websocket)
        print(f"[API] Frontend conectado al stream de alertas: {camera_id}")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        """
        Elimina de la memoria a un cliente que cerró el navegador o perdió conexión.
        """
        if camera_id in self.active_connections:
            try:
                self.active_connections[camera_id].remove(websocket)
                print(f"[API] Frontend desconectado de: {camera_id}")
            except ValueError:
                pass 

    async def broadcast(self, camera_id: str, message: str):
        """
        [Método de Disparo]
        Toma el JSON (alerta de pelea o video listo) generado por el EventManager
        y lo dispara a la velocidad de la luz a todos los navegadores conectados.
        """
        if camera_id in self.active_connections:
            for connection in self.active_connections[camera_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"[API] Error enviando mensaje a cliente en {camera_id}: {e}")