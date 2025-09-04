#!/usr/bin/env python3
"""
GPU memory management utilities for the concurrent RAG system.

This module provides utilities for monitoring and managing GPU memory allocation
to prevent CUDA Out of Memory errors when both Docling and embedding models
compete for GPU resources.
"""
import os
import logging
import contextlib
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUMemoryInfo:
    """GPU memory information."""
    total_memory: int  # bytes
    allocated_memory: int  # bytes
    cached_memory: int  # bytes
    free_memory: int  # bytes
    utilization_percent: float

    def has_sufficient_memory(self, required_mb: int) -> bool:
        """Check if there's sufficient free memory for the requested amount."""
        required_bytes = required_mb * 1024 * 1024
        return self.free_memory >= required_bytes

class GPUMemoryManager:
    """Manages GPU memory allocation and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._torch_available = False
        self._cuda_available = False
        
        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        except ImportError:
            self.logger.warning("PyTorch not available, GPU monitoring disabled")
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for use."""
        return self._cuda_available
    
    def get_memory_info(self, device_id: int = 0) -> Optional[GPUMemoryInfo]:
        """Get current GPU memory information."""
        if not self._cuda_available:
            return None
        
        try:
            import torch
            torch.cuda.set_device(device_id)
            
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            cached_memory = torch.cuda.memory_reserved(device_id)
            free_memory = total_memory - cached_memory
            
            utilization = (allocated_memory / total_memory) * 100
            
            return GPUMemoryInfo(
                total_memory=total_memory,
                allocated_memory=allocated_memory,
                cached_memory=cached_memory,
                free_memory=free_memory,
                utilization_percent=utilization
            )
        except Exception as e:
            self.logger.error(f"Failed to get GPU memory info: {e}")
            return None
    
    def clear_cache(self, device_id: int = 0) -> bool:
        """Clear GPU cache to free up memory."""
        if not self._cuda_available:
            return False
        
        try:
            import torch
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize(device_id)
            
            self.logger.info("GPU cache cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear GPU cache: {e}")
            return False
    
    def estimate_docling_memory_usage(self, file_size_mb: int) -> int:
        """Estimate GPU memory usage for Docling processing in MB."""
        # Conservative estimates based on file size and typical OCR memory usage
        # These are heuristic values based on observation
        if file_size_mb <= 10:
            return 1500  # ~1.5GB for small files
        elif file_size_mb <= 50:
            return 2500  # ~2.5GB for medium files  
        else:
            return 4000  # ~4GB for large files
    
    def estimate_embedding_memory_usage(self, batch_size: int, max_length: int = 384) -> int:
        """Estimate GPU memory usage for embedding model in MB."""
        # Base model memory (GTE-large is ~1.5GB in fp16)
        base_memory = 1500
        
        # Activation memory scales with batch size and sequence length
        # Rough estimate: batch_size * max_length * 2 (bytes) * hidden_dim_factor
        activation_memory = (batch_size * max_length * 2 * 1024) // (1024 * 1024)  # Convert to MB
        
        return base_memory + activation_memory
    
    def can_allocate_for_docling(self, file_size_mb: int, safety_margin_mb: int = 500) -> bool:
        """Check if there's sufficient GPU memory for Docling processing."""
        memory_info = self.get_memory_info()
        if not memory_info:
            return False  # Fallback to CPU if we can't check
        
        required_memory = self.estimate_docling_memory_usage(file_size_mb) + safety_margin_mb
        free_memory_mb = memory_info.free_memory // (1024 * 1024)
        
        can_allocate = free_memory_mb >= required_memory
        self.logger.info(
            f"Docling allocation check: free={free_memory_mb}MB, required={required_memory}MB, "
            f"can_allocate={can_allocate}"
        )
        return can_allocate
    
    def can_allocate_for_embedding(self, batch_size: int, max_length: int = 384, 
                                 safety_margin_mb: int = 500) -> bool:
        """Check if there's sufficient GPU memory for embedding processing."""
        memory_info = self.get_memory_info()
        if not memory_info:
            return False  # Fallback to CPU if we can't check
        
        required_memory = self.estimate_embedding_memory_usage(batch_size, max_length) + safety_margin_mb
        free_memory_mb = memory_info.free_memory // (1024 * 1024)
        
        can_allocate = free_memory_mb >= required_memory
        self.logger.info(
            f"Embedding allocation check: free={free_memory_mb}MB, required={required_memory}MB, "
            f"can_allocate={can_allocate}"
        )
        return can_allocate
    
    @contextlib.contextmanager
    def managed_gpu_allocation(self, operation_name: str, clear_cache_before: bool = True,
                             clear_cache_after: bool = True):
        """Context manager for managed GPU operations with automatic cleanup."""
        if clear_cache_before and self._cuda_available:
            self.logger.info(f"Clearing GPU cache before {operation_name}")
            self.clear_cache()
        
        initial_memory = self.get_memory_info()
        if initial_memory:
            self.logger.info(
                f"Starting {operation_name} - GPU memory: {initial_memory.utilization_percent:.1f}% used"
            )
        
        try:
            yield self
        except Exception as e:
            self.logger.error(f"GPU operation {operation_name} failed: {e}")
            raise
        finally:
            if clear_cache_after and self._cuda_available:
                self.logger.info(f"Clearing GPU cache after {operation_name}")
                self.clear_cache()
            
            final_memory = self.get_memory_info()
            if final_memory:
                self.logger.info(
                    f"Completed {operation_name} - GPU memory: {final_memory.utilization_percent:.1f}% used"
                )
    
    @contextlib.contextmanager
    def force_cpu_context(self):
        """Context manager that forces operations to use CPU by hiding CUDA devices."""
        prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        try:
            self.logger.info("Forcing CPU execution (CUDA hidden)")
            yield
        finally:
            if prev_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
            self.logger.info("Restored CUDA visibility")

# Global instance
gpu_memory_manager = GPUMemoryManager()
