#!/usr/bin/env python3
"""
Example configuration file showing how to use the centralized OpenAI configuration.

Set environment variables to customize OpenAI settings:
- OPENAI_MODEL_NAME: Specific model to use (e.g., "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
- OPENAI_MAX_TOKENS: Maximum tokens per response
- OPENAI_TEMPERATURE: Temperature for response generation (0.0-1.0)
- OPENAI_TIMEOUT_SEC: Timeout in seconds for OpenAI API calls
- OPENAI_MAX_RETRIES: Maximum number of retries for failed calls
- OPENAI_FREQUENCY_PENALTY: Frequency penalty (0.0-2.0)
- OPENAI_PRESENCE_PENALTY: Presence penalty (0.0-2.0)
- OPENAI_TOP_P: Top-p sampling parameter (0.0-1.0)
"""

import os

# Example: Set environment variables for your specific needs
def setup_development_config():
    """Setup development configuration with balanced performance and cost."""
    os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
    os.environ.setdefault("OPENAI_MAX_TOKENS", "1200")
    os.environ.setdefault("OPENAI_TEMPERATURE", "0.1")
    os.environ.setdefault("OPENAI_TIMEOUT_SEC", "90")
    os.environ.setdefault("OPENAI_MAX_RETRIES", "3")

def setup_production_config():
    """Setup production configuration with optimal quality."""
    os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o")
    os.environ.setdefault("OPENAI_MAX_TOKENS", "1500")
    os.environ.setdefault("OPENAI_TEMPERATURE", "0.1")
    os.environ.setdefault("OPENAI_TIMEOUT_SEC", "120")
    os.environ.setdefault("OPENAI_MAX_RETRIES", "3")

def setup_cost_optimized_config():
    """Setup cost-optimized configuration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
    os.environ.setdefault("OPENAI_MAX_TOKENS", "1000")
    os.environ.setdefault("OPENAI_TEMPERATURE", "0.1")
    os.environ.setdefault("OPENAI_TIMEOUT_SEC", "60")
    os.environ.setdefault("OPENAI_MAX_RETRIES", "3")

# Example usage
if __name__ == "__main__":
    # Uncomment the configuration you want to use:
    
    # For development/testing
    setup_development_config()
    
    # For production
    # setup_production_config()
    
    # For cost optimization
    # setup_cost_optimized_config()
    
    # Test the configuration
    from app.shared.openai_config import get_openai_config, get_retrieval_config
    
    config = get_openai_config()
    retrieval_config = get_retrieval_config()
    
    print(f"OpenAI Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Max Tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Timeout: {config.timeout_sec}s")
    print(f"  Max Retries: {config.max_retries}")
    print(f"  Cost per 1K tokens: ${config.cost_per_1k_tokens}")
    
    print(f"\nRetrieval Configuration:")
    print(f"  Similarity Top K: {retrieval_config['similarity_top_k']}")
    print(f"  Final K: {retrieval_config['final_k']}")
    print(f"  Similarity Threshold: {retrieval_config['similarity_threshold']}")
    print(f"  Context Window: {retrieval_config['context_window']}")
