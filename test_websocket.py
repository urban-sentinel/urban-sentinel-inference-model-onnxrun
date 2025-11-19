# test_websocket.py
import asyncio, websockets, json, sys, base64, numpy as np, cv2
from typing import List

CAMERAS_TO_TEST = ["cam_01"]

async def listen_events(uri: str, camera_id: str):
    print(f"--- [Eventos {camera_id}] Conectando a: {uri} ---")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"--- [Eventos {camera_id}] ¡Conectado! Esperando predicciones... ---")
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)

                # 1) Mensaje final del evento (incluye video/log)
                if data.get("type") == "event_complete":
                    print("\n=== EVENTO COMPLETADO ===")
                    print(f"  Cámara: {data.get('camera_id')}")
                    print("  Video:", data.get("video_path"))
                    print("  Log  :", data.get("log_path"))
                    # -> Aquí puedes POSTear a tu backend con 'requests' si quieres.
                    continue

                # 2) Solo mostrar predicciones si superó el umbral (triggered=True)
                if data.get("triggered"):
                    print("\n--- ¡Predicción (umbral superado)! ---")
                    print(f"  Cámara: {data.get('camera_id')}")
                    probs = data.get('probabilities') or {}
                    for k, v in probs.items():
                        print(f"  {k}: {v:.1%}")
                # else: ignorar mensajes “no disparados” para no spamear

    except websockets.exceptions.ConnectionClosed:
        print(f"[Eventos {camera_id}] Conexión cerrada por el servidor.")
    except Exception as e:
        print(f"[Eventos {camera_id}] Error: {e}")

async def listen_video(uri: str, camera_id: str):
    print(f"--- [Video {camera_id}] Conectando a: {uri} ---")
    try:
        async with websockets.connect(uri, max_size=8*1024*1024) as websocket:
            print(f"--- [Video {camera_id}] ¡Conectado! Mostrando 'Video - {camera_id}' ---")
            win = f"Video - {camera_id}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                if data.get("type") != "frame":
                    continue
                b64 = data.get("jpeg_base64")
                if not b64:
                    continue
                jpg = base64.b64decode(b64)
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow(win, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print(f"[Video {camera_id}] Cerrado por teclado.")
    except Exception as e:
        print(f"[Video {camera_id}] Error: {e}")
    finally:
        try: cv2.destroyWindow(f"Video - {camera_id}")
        except: pass

async def main(camera_ids: List[str]):
    tasks = []
    for cam_id in camera_ids:
        events_uri = f"ws://127.0.0.1:8010/ws/{cam_id}"
        frames_uri = f"ws://127.0.0.1:8010/ws/frames/{cam_id}"
        tasks += [
            asyncio.create_task(listen_events(events_uri, cam_id)),
            asyncio.create_task(listen_video(frames_uri, cam_id)),
        ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    cams = CAMERAS_TO_TEST if len(sys.argv) == 1 else sys.argv[1:]
    try:
        asyncio.run(main(cams))
    except KeyboardInterrupt:
        print("\nCerrando clientes...")
        cv2.destroyAllWindows()
