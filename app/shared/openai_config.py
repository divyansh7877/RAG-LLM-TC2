#!/usr/bin/env python3
"""
OpenAI configuration and optimization settings for the RAG system.

This module centralizes all OpenAI-specific configurations and provides
optimized settings for different use cases and environments.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OpenAIModelConfig:
    """Configuration for specific OpenAI models."""
    model_name: str
    max_tokens: int
    temperature: float
    timeout_sec: int
    max_retries: int
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.0
    top_p: float = 0.95
    
    # Cost and performance characteristics
    cost_per_1k_tokens: float = 0.0
    tokens_per_minute_limit: int = 0
    
    def to_openai_params(self) -> Dict[str, Any]:
        """Convert to OpenAI API parameters."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout_sec,
            "max_retries": self.max_retries,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
        }

# Predefined model configurations optimized for RAG tasks
MODEL_CONFIGS = {
    "gpt-4o-mini": OpenAIModelConfig(
        model_name="gpt-4o-mini",
        max_tokens=8000,
        temperature=0.1,
        timeout_sec=90,
        max_retries=3,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        top_p=0.95,
        cost_per_1k_tokens=0.0015,  # Much more cost-effective
        tokens_per_minute_limit=100000  # Higher rate limits
    ),
    
    "gpt-4o": OpenAIModelConfig(
        model_name="gpt-4o",
        max_tokens=1500,
        temperature=0.1,
        timeout_sec=120,
        max_retries=3,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        top_p=0.95,
        cost_per_1k_tokens=0.015,  # Approximate cost
        tokens_per_minute_limit=10000
    ),
    
    "gpt-4-turbo": OpenAIModelConfig(
        model_name="gpt-4-turbo",
        max_tokens=1200,
        temperature=0.1,
        timeout_sec=90,
        max_retries=3,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        top_p=0.95,
        cost_per_1k_tokens=0.01,
        tokens_per_minute_limit=10000
    ),
    
    "gpt-3.5-turbo": OpenAIModelConfig(
        model_name="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.1,
        timeout_sec=60,
        max_retries=4,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        top_p=0.95,
        cost_per_1k_tokens=0.002,
        tokens_per_minute_limit=40000
    ),
    
    # Legacy/fallback models
    "gpt-4": OpenAIModelConfig(
        model_name="gpt-4",
        max_tokens=800,
        temperature=0.1,
        timeout_sec=120,
        max_retries=2,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        top_p=0.95,
        cost_per_1k_tokens=0.03,
        tokens_per_minute_limit=200
    )
}

class OpenAIConfigManager:
    """Manages OpenAI configuration with environment overrides and optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._selected_model = self._determine_model()
        self._config = self._build_config()
    
    def _determine_model(self) -> str:
        """Determine which OpenAI model to use based on environment and availability."""
        # Allow explicit override
        env_model = os.getenv("OPENAI_MODEL_NAME")
        if env_model and env_model in MODEL_CONFIGS:
            self.logger.info(f"Using OpenAI model from environment: {env_model}")
            return env_model
        
        # Check API key availability and model preference
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        # Default priority: gpt-4o-mini > gpt-4o > gpt-4-turbo > gpt-3.5-turbo > gpt-4
        model_priority = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"]
        
        # For now, return the highest priority model
        # In production, you might want to check model availability via API
        selected = model_priority[0]
        self.logger.info(f"Selected OpenAI model: {selected}")
        return selected
    
    def _build_config(self) -> OpenAIModelConfig:
        """Build final configuration with environment overrides."""
        base_config = MODEL_CONFIGS[self._selected_model]
        
        # Apply environment overrides
        env_overrides = {
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", base_config.max_tokens)),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", base_config.temperature)),
            "timeout_sec": int(os.getenv("OPENAI_TIMEOUT_SEC", base_config.timeout_sec)),
            "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", base_config.max_retries)),
            "frequency_penalty": float(os.getenv("OPENAI_FREQUENCY_PENALTY", base_config.frequency_penalty)),
            "presence_penalty": float(os.getenv("OPENAI_PRESENCE_PENALTY", base_config.presence_penalty)),
            "top_p": float(os.getenv("OPENAI_TOP_P", base_config.top_p)),
        }
        
        # Create updated config
        final_config = OpenAIModelConfig(
            model_name=base_config.model_name,
            cost_per_1k_tokens=base_config.cost_per_1k_tokens,
            tokens_per_minute_limit=base_config.tokens_per_minute_limit,
            **env_overrides
        )
        
        self.logger.info(f"OpenAI configuration: {final_config.model_name} with max_tokens={final_config.max_tokens}, "
                        f"temperature={final_config.temperature}, timeout={final_config.timeout_sec}s")
        
        return final_config
    
    def get_config(self) -> OpenAIModelConfig:
        """Get the current OpenAI configuration."""
        return self._config
    
    def get_openai_params(self) -> Dict[str, Any]:
        """Get parameters formatted for OpenAI API."""
        return self._config.to_openai_params()
    
    def get_cost_estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for a query based on token usage."""
        total_tokens = prompt_tokens + completion_tokens
        return (total_tokens / 1000) * self._config.cost_per_1k_tokens
    
    def should_retry_on_error(self, error: Exception) -> tuple[bool, int]:
        """Determine if an error is retryable and suggest delay."""
        error_msg = str(error).lower()
        
        # Rate limit errors
        if "rate limit" in error_msg or "429" in error_msg:
            return True, 30  # Retry after 30 seconds
        
        # Quota exceeded
        if "quota" in error_msg or "insufficient_quota" in error_msg:
            return False, 0  # Don't retry quota issues
        
        # Timeout errors
        if "timeout" in error_msg:
            return True, 10  # Retry after 10 seconds
        
        # Connection errors
        if "connection" in error_msg or "network" in error_msg:
            return True, 5  # Retry after 5 seconds
        
        # Server errors (5xx)
        if "500" in error_msg or "502" in error_msg or "503" in error_msg:
            return True, 15  # Retry after 15 seconds
        
        # Default: don't retry
        return False, 0
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get optimized retrieval configuration for the selected model."""
        # Adjust retrieval parameters based on model capabilities
        if self._config.model_name in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            return {
                "similarity_top_k": 10,
                "final_k": 6,
                "similarity_threshold": 0.6,
                "context_window": 8000  # All these models have good context windows
            }
        elif self._config.model_name == "gpt-3.5-turbo":
            return {
                "similarity_top_k": 8,
                "final_k": 5,
                "similarity_threshold": 0.65,
                "context_window": 3500
            }
        else:  # gpt-4 and others
            return {
                "similarity_top_k": 6,
                "final_k": 4,
                "similarity_threshold": 0.7,
                "context_window": 6000
            }

# Global instance
openai_config_manager = OpenAIConfigManager()

# Convenience functions
def get_openai_config() -> OpenAIModelConfig:
    """Get the current OpenAI configuration."""
    return openai_config_manager.get_config()

def get_openai_params() -> Dict[str, Any]:
    """Get OpenAI parameters for API calls."""
    return openai_config_manager.get_openai_params()

def get_retrieval_config() -> Dict[str, Any]:
    """Get optimized retrieval configuration."""
    return openai_config_manager.get_retrieval_config()

def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost for a query."""
    return openai_config_manager.get_cost_estimate(prompt_tokens, completion_tokens)

def should_retry_error(error: Exception) -> tuple[bool, int]:
    """Check if error should be retried and get delay."""
    return openai_config_manager.should_retry_on_error(error)
