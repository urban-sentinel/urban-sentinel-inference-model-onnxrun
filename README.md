# UrbanSentinel: API Model Documentation

Este repositorio contiene el backend de procesamiento de Inteligencia Artificial para el sistema de videovigilancia **UrbanSentinel**.

El sistema utiliza un enfoque Espacio-Temporal, combinando detección espacial de objetos y rastreo (YOLO) con análisis temporal de secuencias de video (DINOv3) para identificar acciones violentas en tiempo real. Cuenta con una arquitectura altamente concurrente, dinámica (Hot-Plug) y de baja latencia.

---

## 1. Arquitectura de Directorios

El proyecto está diseñado bajo los principios de Arquitectura Hexagonal y SOLID, separando la ingesta, el procesamiento pesado, el almacenamiento y la exposición web.

```text
API_Model/
├── run_app.py                 # Orquestador Principal Dinámico. Inicia subprocesos en caliente, maneja hilo oyente y enlaza colas (IPC).
├── README.md
│
├── core/                      # Configuración y constantes.
│   ├── config.py              # Variables de entorno, rutas, hiperparámetros y umbrales por clase.
│   └── dinov3_cfg.py          # Hiperparámetros de la arquitectura del modelo DINOv3.
│
├── api/                       # Capa de Presentación Web (FastAPI).
│   ├── main.py                # Ensamblador de rutas y ciclo de vida de la API.
│   ├── routes/                
│   │   ├── websocket.py       # Endpoints de streams en tiempo real (alertas y video).
│   │   └── control.py         # Endpoints REST para gestión del ciclo de vida y control de ejecución de cámaras.
│   └── services/              
│       ├── connection_manager.py # Gestión de conexiones y broadcast a sockets activos.
│       ├── event_manager.py   # Orquestador de reglas: decide alertas, graba, inyecta promedios y aplica Garbage Collection.
│       └── prediction_filter.py # Filtro de Media Móvil y Tolerancia con Umbrales Dinámicos OCP.
│
├── ai_engine/                 # Motores de Inteligencia Artificial.
│   ├── models/                
│   │   ├── yolo_spatial.py    # Wrapper ONNX para detección de personas y ByteTrack.
│   │   ├── dinov3_temporal.py # Wrapper PyTorch para inferencia de secuencias.
│   │   └── template/dinov3_model.py # Definición arquitectónica de la red neuronal.
│   └── pipeline/              
│       ├── video_processor.py # Recorte, padding y construcción de Tubelets (Tensores).
│       └── spatial_filter.py  # Filtro EMA para suavizado de cajas (Anti-Jitter).
│
├── workers/                   # Capa de Concurrencia (Multiprocessing).
│   ├── camera_worker.py       # [CPU] Lector, tracking (Ghost Tracking) y envío de recortes.
│   ├── inference_worker.py    # [GPU] Batching dinámico y predicción masiva concurrente.
│   └── recording_worker.py    # [CPU] Proceso I/O aislado para volcado de video al disco.
│
└── io_adapters/               # Adaptadores de Entrada/Salida.
    ├── readers/               # Interfaces para lectura (Archivos, Webcam, RTSP).
    │   ├── base_reader.py, rtsp_reader.py, file_reader.py # (Sincronización FPS añadida).
    └── writers/               # Interfaces para guardar archivos MP4 y JSON.
        ├── base_writer.py, disk_recorder.py

```

---

## 2. Flujo y Transformación de Datos (Pipeline)

El recorrido de la información sigue un pipeline de 5 fases, optimizado con filtros de estabilidad y umbrales matemáticos:

### Fase 1: Ingesta (Sincronizada)

* **Componente:** `io_adapters/readers` -> `camera_worker.py`
* **Datos:** El sistema lee el stream. Para archivos locales, se aplica una sincronización de FPS para simular tiempo real y evitar el procesamiento acelerado.

### Fase 2: Detección Espacial, Suavizado y Clustering

* **Componente:** `yolo_spatial.py` -> `spatial_filter.py` -> `camera_worker.py`
* **Datos:** YOLO detecta personas y las coordenadas pasan por un **Filtro EMA** para eliminar el parpadeo de las cajas. Se aplica **Ghost Tracking** para mantener IDs en frames perdidos. Si las zonas colisionan (margen de 30px), se fusionan en una **Super Caja**.

### Fase 3: Procesamiento Espacio-Temporal (Tubelets)

* **Componente:** `video_processor.py`
* **Datos:** Se acumulan 16 frames. El procesador extrae el historial de la Super Caja suavizada y construye el tensor.
* **Forma:** `torch.Tensor` normalizado en RGB `(3, 16, 224, 224)`.

### Fase 4: Inferencia en Batch (GPU)

* **Componente:** `inference_worker.py` -> `dinov3_temporal.py`
* **Datos:** La GPU procesa el batch (sin importar el número de cámaras activas) y devuelve probabilidades crudas tras la capa Sigmoid.

### Fase 5: Filtrado de Predicción, Cooldown y Disparo

* **Componente:** `prediction_filter.py` -> `event_manager.py`
* **Datos:** Las probabilidades pasan por una **Media Móvil (SMA)**. El sistema extrae la clase violenta dominante en ese instante y la evalúa contra su **umbral específico optimizado (Ej: Golpe vs Patada)**. Si se confirma con tolerancia de frames, se activa la grabación y se aplica un **Cooldown de 5s** para unificar la escena.

---

## 3. Comunicación Backend - Frontend (API)

### A. Gestión del Ciclo de Vida de Cámaras (Orquestación Dinámica)

El backend permite añadir y eliminar fuentes de video en tiempo real sin reiniciar el servidor ni interrumpir la IA.

* **Añadir Nueva Cámara (Hot-Plug):** `POST /control/cameras`
* **Payload:**
```json
{
  "camera_id": "cam_calle_02",
  "camera_type": "file", 
  "path": "ruta/al/video.mp4"
}

```


* *Nota:* Soporta tipos "rtsp", "file" o "webcam".


* **Eliminar Cámara:** `DELETE /control/cameras/{camera_id}`
* Detiene el proceso limpiamente y el `event_manager` ejecuta el Garbage Collection para liberar la RAM.



### B. Control de Ejecución (REST POST)

Controla el flujo de inferencia interno de una cámara que ya está corriendo.

* **Endpoint:** `POST /control/camera`
* **Payload:**
```json
{
  "camera_id": "cam_01",
  "action": "start" 
}

```


*Acciones soportadas:* `"start"`, `"stop"`, `"enable_inference"`, `"disable_inference"`.
* **Respuesta Exitosa (200 OK):**
```json
{ "status": "ok", "camera_id": "cam_01", "action_processed": "start" }

```



### C. Alertas de Violencia (WebSocket)

Canal bidireccional que emite alertas filtradas y notificaciones de archivos listos.

* **Endpoint:** `ws://{host}:8010/ws/{camera_id}`
* **Evento 1: Violencia Detectada:** Incluye la probabilidad suavizada y la clase responsable de detonar la alerta.
```json
{
  "camera_id": "cam_01",
  "group_id": "group_1_2",
  "probabilities": { "Golpe": 0.95, "Peaton": 0.01, "Patada": 0.02, "Forcejeo": 0.02 },
  "clase_dominante": "Golpe",
  "smoothed_prob": 0.895,
  "triggered": true
}

```


* **Evento 2: Zona Segura:** `{"camera_id": "cam_01", "status": "clear", "triggered": false}`
* **Evento 3: Grabación Finalizada:** Notifica el guardado y las rutas finales.

### D. Streaming de Video Crudo (WebSocket)

* **Endpoint:** `ws://{host}:8010/ws/frames/{camera_id}`
* **Payload:** `{"type": "frame", "camera_id": "cam_01", "jpeg_base64": "..."}`

---

## 4. Almacenamiento de Evidencia (Análisis IA)

Al finalizar un evento, el `event_manager` recupera el historial de probabilidades acumuladas durante la grabación, calcula el promedio total de la detección e inyecta esta información en el archivo JSON de log para auditoría.

**Ejemplo de JSON enriquecido:**

```json
{
    "camera_id": "cam_simulada_01",
    "event_start_time": "2026-05-04T18:42:19",
    "video_file": "cam_simulada_01_20260504_184219.mp4",
    "analisis_ia": {
        "metricas_promedio": {
            "Golpe": 0.8951,
            "Peaton": 0.0412,
            "Patada": 0.0120,
            "Forcejeo": 0.0517
        }
    },
    "logs": [ ... ]
}

```
