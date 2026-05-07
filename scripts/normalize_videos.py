import os
import subprocess
import glob
from pathlib import Path

INPUT_DIR = "../test_videos_original" 
OUTPUT_DIR = "../test_videos" 

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30

def check_ffmpeg():
    """Verifica si FFmpeg está instalado y accesible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def normalize_video(input_path: str, output_path: str):
    """
    Usa FFmpeg para normalizar un video.
    Aplica padding negro si el aspect ratio es distinto, fuerza 30 FPS,
    elimina el audio y codifica en H.264 estándar.
    """
    
    video_filter = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease," \
                   f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black," \
                   f"setsar=1"

    cmd = [
        "ffmpeg", 
        "-y",                   
        "-i", input_path,       
        "-vf", video_filter,    
        "-r", str(TARGET_FPS),  
        "-c:v", "libx264",      
        "-preset", "fast",      
        "-profile:v", "main",   
        "-pix_fmt", "yuv420p",  
        "-an",                  
        output_path            
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error procesando {os.path.basename(input_path)}: {e}")
        return False

def main():
    print("=== NORMALIZADOR DE VIDEOS PARA URBANSENTINEL ===")
    
    if not check_ffmpeg():
        print("ERROR CRÍTICO: FFmpeg no está instalado o no está en el PATH de Windows.")
        print("Por favor, instala FFmpeg antes de continuar.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extensiones = ('*.mp4', '*.avi', '*.mkv', '*.mov')
    archivos_crudos = []
    for ext in extensiones:
        archivos_crudos.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not archivos_crudos:
        print(f"No se encontraron videos en '{INPUT_DIR}'.")
        return

    print(f"Encontrados {len(archivos_crudos)} videos. Iniciando conversión a {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS}FPS (Sin Audio)...")
    
    procesados = 0
    for ruta_in in archivos_crudos:
        nombre_archivo = os.path.basename(ruta_in)
        nombre_salida = Path(nombre_archivo).with_suffix('.mp4').name
        ruta_out = os.path.join(OUTPUT_DIR, nombre_salida)

        print(f"  Normalizando: {nombre_archivo} -> ", end="", flush=True)
        
        exito = normalize_video(ruta_in, ruta_out)
        
        if exito:
            print(" OK")
            procesados += 1

    print("\n¡Proceso Finalizado!")
    print(f"Videos listos para la API guardados en: {OUTPUT_DIR}")
    print("En tu run_app.py, apunta CAMERAS_TO_RUN a esta carpeta.")

if __name__ == "__main__":
    main()