# ver_webcam.py
import cv2

def abrir_camara(index=0, width=1280, height=720, fps=30):
    # En Windows, CAP_DSHOW acelera y evita bloqueos
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara (índice {index}).")

    # Intenta fijar resolución y fps (algunas webcams ignoran estos valores)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,         fps)

    print(f"Cámara abierta: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    return cap

def main():
    cap = abrir_camara(index=0, width=1280, height=720, fps=30)  # cambia index a 1/2 si tienes varias
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️  Frame no válido, reintentando…")
                continue

            cv2.imshow("Logitech C925e (ESC o Q para salir)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC o q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
