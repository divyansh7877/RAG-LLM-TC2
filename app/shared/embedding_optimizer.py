#!/usr/bin/env python3
"""
Embedding optimization utilities to improve performance and avoid warnings.
"""
import os
import logging
from typing import Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

def set_optimal_threading_environment():
    """Set environment variables for optimal threading performance."""
    # Prevent OpenBLAS threading issues that cause warnings
    threading_vars = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", 
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
    }
    
    for var, value in threading_vars.items():
        os.environ[var] = value
        
    logger.debug(f"Set threading environment variables: {threading_vars}")

def configure_torch_for_cpu():
    """Configure PyTorch for optimal CPU performance."""
    try:
        import torch
        
        # Set single thread for CPU inference to avoid conflicts
        torch.set_num_threads(1)
        
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        
        # Use deterministic algorithms for reproducibility
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
            
        logger.debug("Configured PyTorch for optimal CPU performance")
        
    except ImportError:
        logger.warning("PyTorch not available, skipping torch configuration")

def create_optimized_embedding_model(
    model_name: str = "./models/gte-large-en-v1.5",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512
) -> HuggingFaceEmbedding:
    """
    Create an optimized HuggingFace embedding model.
    
    Args:
        model_name: Path or name of the embedding model
        device: Device to use ('cpu' or 'cuda')
        batch_size: Batch size for embedding generation
        max_length: Maximum sequence length
        
    Returns:
        Optimized HuggingFaceEmbedding instance
    """
    # Set optimal environment
    set_optimal_threading_environment()
    
    if device == "cpu":
        configure_torch_for_cpu()
    
    # Create embedding model with optimized settings
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        trust_remote_code=True,
        embed_batch_size=batch_size,
        max_length=max_length,
        # Additional optimizations
        normalize=True,  # Normalize embeddings for better similarity search
        query_instruction="",  # No special query instruction needed
        text_instruction="",   # No special text instruction needed
    )
    
    logger.info(f"Created optimized embedding model: {model_name} on {device}")
    logger.info(f"Settings: batch_size={batch_size}, max_length={max_length}")
    
    return embed_model

def get_embedding_model_singleton(
    model_name: str = "./models/gte-large-en-v1.5",
    device: str = "cpu"
) -> HuggingFaceEmbedding:
    """
    Get a singleton instance of the embedding model for reuse.
    
    This avoids reloading the model multiple times and improves performance.
    """
    # Use a simple class attribute to store the singleton
    if not hasattr(get_embedding_model_singleton, '_model'):
        get_embedding_model_singleton._model = None
        get_embedding_model_singleton._model_name = None
        get_embedding_model_singleton._device = None
    
    # Check if we need to create a new model
    if (get_embedding_model_singleton._model is None or 
        get_embedding_model_singleton._model_name != model_name or
        get_embedding_model_singleton._device != device):
        
        logger.info(f"Creating new embedding model singleton: {model_name} on {device}")
        get_embedding_model_singleton._model = create_optimized_embedding_model(
            model_name=model_name,
            device=device,
            batch_size=8,  # Conservative batch size for singleton
            max_length=512
        )
        get_embedding_model_singleton._model_name = model_name
        get_embedding_model_singleton._device = device
    
    return get_embedding_model_singleton._model

def optimize_for_batch_processing(batch_size: int) -> dict:
    """
    Get optimal settings for batch processing based on batch size.
    
    Args:
        batch_size: Number of items to process in batch
        
    Returns:
        Dictionary with optimal settings
    """
    if batch_size <= 8:
        return {
            "embed_batch_size": min(batch_size, 4),
            "max_length": 512,
            "node_batch_size": 8
        }
    elif batch_size <= 32:
        return {
            "embed_batch_size": 8,
            "max_length": 512,
            "node_batch_size": 16
        }
    else:
        return {
            "embed_batch_size": 16,
            "max_length": 384,  # Shorter sequences for large batches
            "node_batch_size": 32
        }

def cleanup_embedding_resources():
    """Clean up embedding model resources."""
    if hasattr(get_embedding_model_singleton, '_model'):
        if get_embedding_model_singleton._model is not None:
            # Clear the model from memory
            del get_embedding_model_singleton._model
            get_embedding_model_singleton._model = None
            get_embedding_model_singleton._model_name = None
            get_embedding_model_singleton._device = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Cleaned up embedding model resources")