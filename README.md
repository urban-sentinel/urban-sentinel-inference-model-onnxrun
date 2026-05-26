# UrbanSentinel: API Model Documentation

Este repositorio contiene el backend de procesamiento de Inteligencia Artificial para el sistema de videovigilancia **UrbanSentinel**.

El sistema utiliza un enfoque Espacio-Temporal, combinando detección espacial de objetos y rastreo (YOLO) con análisis temporal de secuencias de video (DINOv3) para identificar acciones violentas en tiempo real. Cuenta con una arquitectura altamente concurrente, dinámica (Hot-Plug), de baja latencia y con optimización de grado industrial (Zero-Copy, decodificación PyAV y compilación JIT).

---

## 1. Arquitectura de Directorios

El proyecto está diseñado bajo los principios de Arquitectura Hexagonal y SOLID, separando la ingesta, el procesamiento pesado, el almacenamiento y la exposición web.

```text
API_Model/
├── run_app.py                 # Orquestador Principal Dinámico. Inicia subprocesos en caliente, enlaza colas (IPC), maneja motor uvloop (Linux) y administra el Pool de Memoria Compartida.
├── README.md
│
├── core/                      # Configuración y constantes.
│   ├── config.py              # Variables de entorno, rutas, hiperparámetros y configuración del Pool de Memoria (Zero-Copy).
│   └── dinov3_cfg.py          # Hiperparámetros de la arquitectura del modelo DINOv3.
│
├── api/                       # Capa de Presentación Web (FastAPI).
│   ├── main.py                # Ensamblador de rutas y ciclo de vida de la API.
│   ├── routes/                
│   │   ├── websocket.py       # Endpoints de streams en tiempo real (alertas y video).
│   │   └── control.py         # Endpoints REST para gestión del ciclo de vida y control de ejecución de cámaras.
│   └── services/              
│       ├── connection_manager.py # Gestión de conexiones y broadcast a sockets activos.
│       ├── event_manager.py   # Orquestador de reglas: decide alertas, graba, inyecta promedios (Serialización en Rust con orjson) y aplica Garbage Collection.
│       └── prediction_filter.py # Filtro de Media Móvil y Tolerancia con Umbrales Dinámicos OCP.
│
├── ai_engine/                 # Motores de Inteligencia Artificial.
│   ├── models/                
│   │   ├── yolo_spatial.py    # Wrapper ONNX para detección de personas y ByteTrack.
│   │   ├── dinov3_temporal.py # Wrapper PyTorch para inferencia de secuencias (Auto-optimizado con torch.compile).
│   │   └── template/dinov3_model.py # Definición arquitectónica de la red neuronal.
│   └── pipeline/              
│       ├── shared_buffer_manager.py # Pool de memoria RAM de tamaño fijo (Estacionamiento) para transferencias Zero-Copy.
│       ├── video_processor.py # Recorte, padding e inyección directa de recortes a la Memoria Compartida (NumPy).
│       └── spatial_filter.py  # Filtro EMA para suavizado de cajas (Anti-Jitter).
│       └── track_buffer_manager.py #NUEVO
│
├── workers/                   # Capa de Concurrencia (Multiprocessing).
│   ├── camera_worker.py       # [CPU] Tracking (Ghost Tracking) y envío de 'tickets' de memoria (Eliminación de Pickling).
│   ├── inference_worker.py    # [GPU] Lectura Zero-Copy, Batching dinámico y predicción masiva concurrente.
│   └── recording_worker.py    # [CPU] Proceso I/O aislado para volcado de video asíncrono al disco.
│
└── io_adapters/               # Adaptadores de Entrada/Salida.
    ├── readers/               # Interfaces para lectura. Implementación PyAV (multithreading/TCP) para RTSP/Archivos (Bajo consumo CPU) y OpenCV para Webcam.
    │   ├── base_reader.py, rtsp_reader.py, file_reader.py, webcam_reader.py
    └── writers/               # Interfaces para compresión de video.
        ├── base_writer.py, disk_recorder.py # Grabación asíncrona H.264 con PyAV (preset ultrafast) para minimizar impacto térmico.


```

---

## 2. Flujo y Transformación de Datos (Pipeline)

El recorrido de la información sigue un pipeline de 5 fases, optimizado con filtros de estabilidad y umbrales matemáticos:

### Fase 1: Ingesta (Sincronizada y Optimizada)

* **Componente:** `io_adapters/readers` -> `camera_worker.py`
* **Datos:** El sistema lee el stream utilizando el motor de **PyAV (FFmpeg)** para minimizar el uso de CPU y garantizar tolerancia a caídas de red TCP. Para archivos locales, se aplica una sincronización de FPS para simular tiempo real y evitar el procesamiento acelerado.

### Fase 2: Detección Espacial, Suavizado y Clustering

* **Componente:** `yolo_spatial.py` -> `spatial_filter.py` -> `camera_worker.py`
* **Datos:** YOLO detecta personas y las coordenadas pasan por un **Filtro EMA** para eliminar el parpadeo de las cajas. Se aplica **Ghost Tracking** para mantener IDs en frames perdidos. Si las zonas colisionan (margen de 30px), se fusionan en una **Super Caja**.

### Fase 3: Procesamiento Espacio-Temporal (Zero-Copy)

* **Componente:** `video_processor.py` -> `shared_buffer_manager.py`
* **Datos:** Se acumulan 16 frames. El procesador extrae el historial de la Super Caja y lo inyecta directamente en un bloque estático de **Memoria Compartida en RAM**. Para evitar el cuello de botella de la serialización (Pickling), la cola multiproceso solo transporta un "ticket" o índice del bloque, reduciendo la latencia de IPC a 0ms.

### Fase 4: Inferencia en Batch (GPU)

* **Componente:** `inference_worker.py` -> `dinov3_temporal.py`
* **Datos:** El worker de IA lee los tickets, extrae instantáneamente los datos de la memoria pública a la VRAM, arma el Batch (sin importar el número de cámaras activas) y libera la RAM compartida. La predicción se acelera automáticamente mediante `torch.compile` (si el entorno lo permite) y devuelve probabilidades crudas tras la capa Sigmoid.

### Fase 5: Filtrado de Predicción, Cooldown y Disparo

* **Componente:** `prediction_filter.py` -> `event_manager.py`
* **Datos:** Las probabilidades pasan por una **Media Móvil (SMA)**. El sistema extrae la clase violenta dominante en ese instante y la evalúa contra su **umbral específico optimizado (Ej: Golpe vs Patada)**. Si se confirma con tolerancia de frames, se activa la grabación asíncrona en disco y se aplica un **Cooldown de 5s** para unificar la escena.

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

Al finalizar un evento, el `event_manager` recupera el historial de probabilidades acumuladas durante la grabación, calcula el promedio total de la detección e inyecta esta información en el archivo JSON de log de forma ultrarrápida (usando motor de serialización asíncrono) para auditoría.

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

