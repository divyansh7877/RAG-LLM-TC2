#!/usr/bin/env python3
"""
Embedding model management and optimization with advanced GPU memory management.

This module provides a thread-safe singleton for the embedding model with intelligent
GPU memory allocation to prevent CUDA OOM errors when competing with other GPU processes.
"""
import os
import logging
import threading
from typing import Dict, Optional, Tuple

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import config
from .gpu_memory_manager import gpu_memory_manager

logger = logging.getLogger(__name__)

# --- Thread-safe Singleton for Embedding Model ---
class EmbeddingModelSingleton:
    _instance: HuggingFaceEmbedding = None
    _device: Optional[str] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str, device: Optional[str] = None) -> HuggingFaceEmbedding:
        """
        Get the singleton instance of the embedding model with intelligent device selection.
        Initializes the model on the first call with GPU memory management.
        """
        resolved_device = cls._resolve_optimal_device(device)
        
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info(f"Initializing embedding model singleton: {model_name} on {resolved_device}")
                    cls._instance = cls._create_optimized_model(model_name, resolved_device)
                    cls._device = resolved_device
        return cls._instance
    
    @classmethod
    def _resolve_optimal_device(cls, preferred_device: Optional[str] = None) -> str:
        """
        Resolve the optimal device for the embedding model based on memory availability.
        """
        if preferred_device == "cpu":
            return "cpu"
        
        # Check if GPU is available and has sufficient memory
        if gpu_memory_manager.is_gpu_available():
            # Use conservative batch size for memory estimation
            batch_size = int(os.getenv("EMBED_BATCH_SIZE", "16"))
            max_length = int(os.getenv("EMBED_MAX_LENGTH", "384"))
            
            if gpu_memory_manager.can_allocate_for_embedding(batch_size, max_length):
                logger.info("GPU has sufficient memory for embedding model")
                return "cuda"
            else:
                logger.warning("Insufficient GPU memory for embedding model, falling back to CPU")
                return "cpu"
        else:
            logger.info("GPU not available, using CPU for embedding model")
            return "cpu"
    
    @classmethod
    def clear_instance(cls):
        """Clear the singleton instance to force reinitialization (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                logger.info("Clearing embedding model singleton instance")
                cls._instance = None
                cls._device = None
                # Clear GPU cache if we were using GPU
                gpu_memory_manager.clear_cache()

    @staticmethod
    def _create_optimized_model(model_name: str, device: str) -> HuggingFaceEmbedding:
        """
        Create an optimized HuggingFace embedding model with GPU memory management.
        - FP16 on CUDA to reduce VRAM usage
        - Adaptive batch sizes based on available memory
        - Proper memory cleanup and monitoring
        """
        # Threading/env optimizations
        threading_vars = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for var, value in threading_vars.items():
            os.environ[var] = value

        model_kwargs = {}
        if device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    # Enable optimizations for GPU
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    model_kwargs["torch_dtype"] = torch.float16
                    logger.info("Enabled GPU optimizations (FP16, TF32)")
            except Exception as e:
                logger.warning(f"Failed to enable GPU optimizations: {e}")
        else:
            # CPU optimizations
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_grad_enabled(False)
                logger.info("Enabled CPU optimizations")
            except Exception:
                logger.debug("PyTorch CPU optimization skipped (torch not available).")

        # Adaptive batch size and sequence length based on device and memory
        embed_batch_size, max_length = EmbeddingModelSingleton._get_optimal_params(device)
        
        logger.info(f"Creating embedding model with batch_size={embed_batch_size}, max_length={max_length}")
        
        # Clear GPU cache before loading model
        if device != "cpu":
            gpu_memory_manager.clear_cache()
        
        try:
            model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                trust_remote_code=True,
                embed_batch_size=embed_batch_size,
                max_length=max_length,
                normalize=True,
                model_kwargs=model_kwargs or None,
            )
            
            # Log memory usage after model creation
            if device != "cpu":
                memory_info = gpu_memory_manager.get_memory_info()
                if memory_info:
                    logger.info(f"Model loaded - GPU memory usage: {memory_info.utilization_percent:.1f}%")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create embedding model on {device}: {e}")
            if device != "cpu":
                logger.info("Attempting CPU fallback for embedding model")
                # Clear GPU cache and try CPU
                gpu_memory_manager.clear_cache()
                return EmbeddingModelSingleton._create_optimized_model(model_name, "cpu")
            raise
    
    @staticmethod
    def _get_optimal_params(device: str) -> Tuple[int, int]:
        """Get optimal batch size and max length based on device and available memory."""
        # Start with environment overrides if available
        env_batch_size = os.getenv("EMBED_BATCH_SIZE")
        env_max_length = os.getenv("EMBED_MAX_LENGTH")
        
        if env_batch_size and env_max_length:
            return int(env_batch_size), int(env_max_length)
        
        if device == "cpu":
            # CPU defaults - can be more generous as we don't have VRAM constraints
            return 32, 512
        
        # GPU - adaptive based on available memory
        memory_info = gpu_memory_manager.get_memory_info()
        if not memory_info:
            # Conservative defaults if we can't check memory
            return 8, 256
        
        free_memory_gb = memory_info.free_memory / (1024 ** 3)
        
        if free_memory_gb >= 6:  # Plenty of memory
            return 16, 384
        elif free_memory_gb >= 4:  # Moderate memory
            return 12, 320
        elif free_memory_gb >= 2:  # Limited memory
            return 8, 256
        else:  # Very limited memory
            return 4, 128

def get_embedding_model(model_name: str, device: Optional[str] = None) -> HuggingFaceEmbedding:
    """Public function to access the embedding model singleton with GPU memory management."""
    return EmbeddingModelSingleton.get_instance(model_name, device)

def clear_embedding_model():
    """Clear the embedding model singleton to free up GPU memory."""
    EmbeddingModelSingleton.clear_instance()

def optimize_for_batch_processing(node_count: int, force_conservative: bool = False) -> Dict[str, int]:
    """
    Get optimal settings for batch processing based on the number of nodes and available GPU memory.
    Now includes GPU memory-aware optimization.
    
    Args:
        node_count: Number of nodes to process
        force_conservative: If True, use conservative settings regardless of memory availability
    
    Returns:
        Dictionary with optimal batch sizes
    """
    # Check available GPU memory if using GPU
    memory_info = gpu_memory_manager.get_memory_info()
    
    # Base recommendations
    if node_count <= 32:
        base_embed_batch = 8
        base_node_batch = 32
    elif node_count <= 128:
        base_embed_batch = 12
        base_node_batch = 64
    elif node_count <= 512:
        base_embed_batch = 16
        base_node_batch = 128
    else:
        base_embed_batch = 20
        base_node_batch = 256
    
    # Adjust based on GPU memory availability
    if memory_info and not force_conservative:
        free_memory_gb = memory_info.free_memory / (1024 ** 3)
        
        if free_memory_gb < 2:  # Very limited memory
            embed_batch_size = max(4, base_embed_batch // 4)
            node_batch_size = max(16, base_node_batch // 4)
            logger.info(f"Very limited GPU memory ({free_memory_gb:.1f}GB), using conservative batch sizes")
        elif free_memory_gb < 4:  # Limited memory
            embed_batch_size = max(6, base_embed_batch // 2)
            node_batch_size = max(24, base_node_batch // 2)
            logger.info(f"Limited GPU memory ({free_memory_gb:.1f}GB), using reduced batch sizes")
        else:  # Sufficient memory
            embed_batch_size = base_embed_batch
            node_batch_size = base_node_batch
            logger.info(f"Sufficient GPU memory ({free_memory_gb:.1f}GB), using standard batch sizes")
    else:
        # Conservative defaults for CPU or when memory info unavailable
        embed_batch_size = max(4, base_embed_batch // 2)
        node_batch_size = max(16, base_node_batch // 2)
        logger.info("Using conservative batch sizes (CPU or memory info unavailable)")
    
    return {
        "embed_batch_size": embed_batch_size,
        "node_batch_size": node_batch_size
    }
