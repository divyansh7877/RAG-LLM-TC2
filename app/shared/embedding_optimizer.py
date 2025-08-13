#!/usr/bin/env python3
"""
Embedding model management and optimization.

This module provides a thread-safe singleton for the embedding model to ensure it is
loaded only once per worker process, which is critical for performance and memory usage.
"""
import os
import logging
import threading
from typing import Dict

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

# --- Thread-safe Singleton for Embedding Model ---
class EmbeddingModelSingleton:
    _instance: HuggingFaceEmbedding = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str, device: str) -> HuggingFaceEmbedding:
        """
        Get the singleton instance of the embedding model.
        Initializes the model on the first call.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info(f"Initializing embedding model singleton: {model_name} on {device}")
                    cls._instance = cls._create_optimized_model(model_name, device)
        return cls._instance

    @staticmethod
    def _create_optimized_model(model_name: str, device: str) -> HuggingFaceEmbedding:
        """
        Create an optimized HuggingFace embedding model with specific settings.
        """
        # Set environment variables for optimal threading performance
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

        # Configure PyTorch for CPU if needed
        if device == "cpu":
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_grad_enabled(False)
            except ImportError:
                logger.warning("PyTorch not available, skipping CPU optimization.")

        # Create the model with optimized settings
        return HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            trust_remote_code=True,
            embed_batch_size=16,  # A reasonable default batch size
            max_length=512,
            normalize=True,
        )

def get_embedding_model(model_name: str, device: str) -> HuggingFaceEmbedding:
    """Public function to access the embedding model singleton."""
    return EmbeddingModelSingleton.get_instance(model_name, device)

def optimize_for_batch_processing(node_count: int) -> Dict[str, int]:
    """
    Get optimal settings for batch processing based on the number of nodes.
    This helps manage memory and improve throughput.
    
    Args:
        node_count: The total number of nodes to be processed.
        
    Returns:
        A dictionary with optimal settings for embedding and node processing.
    """
    if node_count <= 32:
        return {"embed_batch_size": 8, "node_batch_size": 32}
    elif node_count <= 128:
        return {"embed_batch_size": 16, "node_batch_size": 64}
    else:
        return {"embed_batch_size": 32, "node_batch_size": 128}
