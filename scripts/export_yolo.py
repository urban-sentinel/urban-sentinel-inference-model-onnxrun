import os
import shutil
from ultralytics import YOLO

def main():
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "ai_engine", "weights")
    
    os.makedirs(weights_dir, exist_ok=True)
    
    final_onnx_path = os.path.join(weights_dir, "yolo11s.onnx")

    print("Iniciando el proceso de preparación de YOLO11 Small...")
    print(f"Carpeta de destino final: {weights_dir}")
    print("\n⬇Descargando pesos oficiales de YOLO11s (PyTorch)...")

    model = YOLO("yolo11s.pt") 

    print("\nExportando modelo a formato ONNX (Resolución: 640x640)...")

    exported_file_path = model.export(format="onnx", imgsz=640)

    print("\nMoviendo el modelo a la carpeta del Motor de IA...")
    
    if os.path.exists(final_onnx_path):
        os.remove(final_onnx_path)
        
    shutil.move(exported_file_path, final_onnx_path)
    
    if os.path.exists("yolo11s.pt"):
        os.remove("yolo11s.pt")
        print("Archivo temporal 'yolo11s.pt' eliminado.")

    print(f"\n¡Éxito! Modelo ONNX listo para producción en: {final_onnx_path}")

if __name__ == "__main__":
    main()