# OpenAI Integration and Rate Limiting Guide

This document covers the OpenAI integration, including the new GPT-4o mini configuration, rate limiting, and quota management features.

## Overview

Your RAG system now uses **GPT-4o mini** as the default OpenAI model, which provides:
- 🎯 **10x lower cost** compared to GPT-4o ($0.0015 vs $0.015 per 1K tokens)
- 🚀 **Higher rate limits** (100K tokens/min vs 10K tokens/min)
- 🧠 **Similar performance** for RAG tasks
- ⚡ **Faster response times**

## Configuration

### Environment Variables

Set these environment variables to customize OpenAI behavior:

```bash
# OpenAI API configuration
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL_NAME="gpt-4o-mini"  # Default model
export OPENAI_MAX_TOKENS="1500"         # Max tokens per response
export OPENAI_TEMPERATURE="0.1"         # Response consistency (0.0-1.0)
export OPENAI_TIMEOUT_SEC="90"          # Request timeout
export OPENAI_MAX_RETRIES="3"           # Retry attempts
```

### Available Models

The system supports these models (in priority order):
1. **gpt-4o-mini** (default) - Cost-effective, fast
2. **gpt-4o** - Premium quality, higher cost
3. **gpt-4-turbo** - Balanced performance
4. **gpt-3.5-turbo** - Legacy fallback
5. **gpt-4** - Legacy premium

## Rate Limiting and Quota Management

### Automatic Protection

The system now includes intelligent rate limiting:
- ⏱️ **Automatic request spacing** to stay within API limits
- 🔄 **Smart retry logic** for rate limit errors
- ❌ **Quota exhaustion detection** with graceful fallbacks
- 📊 **Usage tracking** and cost estimation

### Monitoring Tools

Use the CLI monitoring tool to track usage:

```bash
# Check current configuration
python scripts/monitor_openai_usage.py config

# View quota status
python scripts/monitor_openai_usage.py quota

# Show 24-hour usage statistics
python scripts/monitor_openai_usage.py stats

# Show 48-hour statistics
python scripts/monitor_openai_usage.py stats --hours 48

# Estimate cost for 5000 tokens
python scripts/monitor_openai_usage.py cost --tokens 5000

# Reset quota status (if needed)
python scripts/monitor_openai_usage.py reset
```

## Cost Optimization

### GPT-4o Mini Benefits

With GPT-4o mini, your costs are dramatically reduced:

| Model | Cost per 1K tokens | 100K tokens | 1M tokens |
|-------|-------------------|-------------|-----------|
| GPT-4o | $0.015 | $1.50 | $15.00 |
| **GPT-4o mini** | **$0.0015** | **$0.15** | **$1.50** |

### Usage Examples

**Typical query costs with GPT-4o mini:**
- Simple query (500 tokens): $0.0008
- Complex query (1500 tokens): $0.0023
- Document summary (3000 tokens): $0.0045

## Error Handling

### Quota Exceeded

When quota is exceeded, the system:
1. 🛑 **Stops new requests** automatically
2. 📝 **Logs detailed error** information
3. 👤 **Returns user-friendly** error messages
4. ⏰ **Tracks reset time** for automatic recovery

### Rate Limits

For rate limit errors, the system:
1. 🔄 **Automatically retries** with exponential backoff
2. ⏱️ **Implements intelligent delays** between requests
3. 📊 **Tracks usage patterns** to prevent future issues

## Configuration Examples

### Development Setup
```python
# Balanced performance and cost
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["OPENAI_MAX_TOKENS"] = "1200"
os.environ["OPENAI_TEMPERATURE"] = "0.1"
```

### Production Setup
```python
# Maximum quality (higher cost)
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
os.environ["OPENAI_MAX_TOKENS"] = "1500"
os.environ["OPENAI_TEMPERATURE"] = "0.1"
```

### Cost-Optimized Setup
```python
# Minimal cost while maintaining quality
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
os.environ["OPENAI_MAX_TOKENS"] = "1000"
os.environ["OPENAI_TEMPERATURE"] = "0.1"
```

## Troubleshooting

### Quota Issues

If you see "quota exceeded" errors:

1. **Check your OpenAI billing**: Visit https://platform.openai.com/account/billing
2. **Add credits**: Ensure sufficient balance for your usage
3. **Reset quota tracking**: `python scripts/monitor_openai_usage.py reset`
4. **Monitor usage**: Use the monitoring tools to track patterns

### Rate Limit Issues

For rate limit problems:

1. **Check current limits**: OpenAI dashboard shows your tier limits
2. **Reduce concurrency**: Limit simultaneous queries if needed
3. **Monitor patterns**: Use the usage stats to identify spikes

### Model Errors

If a model is unavailable:
1. **Check model name**: Ensure correct spelling in environment variables
2. **Try fallback**: System automatically falls back to available models
3. **Update API key**: Ensure your key has access to the model

## Best Practices

### Cost Management
- 📊 **Monitor usage regularly** using the CLI tools
- 🎯 **Use appropriate models** for your use case
- ⚡ **Enable caching** to reduce duplicate requests
- 🔄 **Set reasonable token limits** for your queries

### Performance
- 🚀 **Use GPT-4o mini** for most RAG tasks
- 📝 **Optimize prompts** to reduce token usage
- ⏱️ **Set appropriate timeouts** for your network
- 🔄 **Implement retries** for reliability

### Monitoring
- 📈 **Track daily usage** to identify trends
- 🔔 **Set up alerts** for unusual usage patterns  
- 💰 **Monitor costs** regularly
- 📊 **Analyze patterns** to optimize performance

## Integration with Your RAG System

The OpenAI integration is seamlessly integrated into your RAG pipeline:

1. **Query Engine Factory** automatically uses the configured model
2. **Rate Limiting** is applied transparently during query processing
3. **Cost Tracking** provides real-time usage and cost estimates
4. **Error Handling** provides graceful degradation when issues occur

Your existing API endpoints and workflows require no changes - everything works automatically with the new configuration system.
