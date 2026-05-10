import numpy as np
from multiprocessing import shared_memory, Queue
from queue import Empty
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.config import config

class SharedBufferManager:
    """
    Administrador de Memoria Compartida (Shared Memory Pool).
    Responsable de crear, asignar y destruir bloques estáticos de RAM 
    para permitir transferencias Zero-Copy entre procesos.
    """
    def __init__(self):
        self.num_blocks = config.SHM_MAX_BLOCKS
        self.shape = config.SHM_TENSOR_SHAPE
        self.dtype = np.dtype(config.SHM_TENSOR_DTYPE)
        
        self.block_size = int(np.prod(self.shape)) * self.dtype.itemsize
        self.prefix = config.SHM_PREFIX
        
        self.shm_blocks = []
        
        self.free_queue = Queue()

        print(f"[SharedMemory] Asignando {self.num_blocks} bloques de {self.block_size / (1024*1024):.2f} MB cada uno...")
        
        for i in range(self.num_blocks):
            block_name = f"{self.prefix}{i}"
            try:
                try:
                    existing_shm = shared_memory.SharedMemory(name=block_name)
                    existing_shm.unlink()
                except FileNotFoundError:
                    pass
                
                shm = shared_memory.SharedMemory(name=block_name, create=True, size=self.block_size)
                self.shm_blocks.append(shm)
                
                self.free_queue.put(i)
                
            except Exception as e:
                print(f"[SharedMemory] Error crítico creando bloque {i}: {e}")
                self.cleanup()
                raise

        print(f"[SharedMemory] Pool inicializado. RAM estática reservada: {(self.num_blocks * self.block_size) / (1024*1024):.2f} MB")

    def get_free_block(self):
        """
        La cámara llama a este método. Obtiene el índice del próximo bloque libre.
        Si el estacionamiento está lleno, retorna (None, None).
        """
        try:
            index = self.free_queue.get(timeout=0.01)
            block_name = f"{self.prefix}{index}"
            return index, block_name
        except Empty:
            return None, None

    def release_block(self, index: int):
        """
        La GPU (o el Orquestador) llama a este método. 
        Devuelve el ticket a la cola para que otra cámara pueda usar este bloque.
        """
        if 0 <= index < self.num_blocks:
            self.free_queue.put(index)

    def cleanup(self):
        """
        Destruye la memoria y se la devuelve al Sistema Operativo. 
        Vital para no causar "Memory Leaks" masivos.
        """
        print("[SharedMemory] Destruyendo bloques de memoria y limpiando RAM...")
        for shm in self.shm_blocks:
            try:
                shm.close()
                shm.unlink() 
            except Exception:
                pass
        self.shm_blocks.clear()
        
        while not self.free_queue.empty():
            try:
                self.free_queue.get_nowait()
            except Empty:
                break
                
        print("[SharedMemory] RAM liberada con éxito.")