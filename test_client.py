import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import ctypes  

CAMERA_ID = "cam_24"  
API_HOST = "ws://127.0.0.1:8010"

current_boxes = {}

async def listen_alerts():
    """Conecta al WebSocket de eventos, escucha violencia y coordenadas (boxes)."""
    global current_boxes
    uri = f"{API_HOST}/ws/{CAMERA_ID}"
    print(f"📡 [Alertas] Conectando a {uri}...")
    
    async with websockets.connect(uri) as ws:
        print("✅ [Alertas] ¡Conectado! Esperando detecciones...\n")
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                if data.get("type") == "boxes":
                    current_boxes = data.get("data", {})
                    continue

                if data.get("type") == "event_complete":
                    print(f"\n💾 [GRABACIÓN LISTA] MP4 guardado en: {data.get('video_file')}")
                    continue

                if data.get("triggered"):
                    group_id = data.get("group_id")
                    probs = data.get("probabilities", {})
                    clase_dom = data.get("clase_dominante")
                    
                    print(f"🚨 [ALERTA DE VIOLENCIA] - Caja: {group_id}")
                    print(f"   ┣ Acción Dominante: {clase_dom}")
                    for accion, prob in probs.items():
                        if prob > 0.01:
                            print(f"   ┣ {accion}: {prob:.1%}")
                    print("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                elif data.get("status") == "clear":
                    print("🟢 [ESTADO] Interacciones pacíficas o calle vacía.")
                    current_boxes = {} 
                    
            except Exception as e:
                print(f"⚠️ [Alertas] Error: {e}")

async def listen_video():
    """Conecta al WebSocket de video, dibuja las cajas de la memoria global y muestra."""
    global current_boxes
    uri = f"{API_HOST}/ws/frames/{CAMERA_ID}"
    print(f"🎥 [Video] Conectando a {uri}...")
    
    window_name = f"Monitor - {CAMERA_ID}"
    WIN_WIDTH = 1280
    WIN_HEIGHT = 720
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIN_WIDTH, WIN_HEIGHT)
    
    try:
        user32 = ctypes.windll.user32
        center_x = (user32.GetSystemMetrics(0) - WIN_WIDTH) // 2
        center_y = (user32.GetSystemMetrics(1) - WIN_HEIGHT) // 2
        cv2.moveWindow(window_name, center_x, center_y)
    except: pass
    
    try:
        async with websockets.connect(uri, max_size=8*1024*1024) as ws:
            print("✅ [Video] ¡Conectado! Mostrando stream...\n")
            
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                b64_img = data.get("jpeg_base64")
                if not b64_img: continue
                    
                img_bytes = base64.b64decode(b64_img)
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    for group_id, box in current_boxes.items():
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, str(group_id), (x1, max(0, y1-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("🛑 Cerrando monitor...")
                        break
                        
    except Exception as e:
        print(f"⚠️ [Video] Error: {e}")
    finally:
        cv2.destroyAllWindows()

async def main():
    await asyncio.gather(listen_video(), listen_alerts())

if __name__ == "__main__":
    print("=== INICIANDO CLIENTE DE PRUEBA ===")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrueba finalizada por el usuario.")