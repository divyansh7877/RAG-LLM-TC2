#!/usr/bin/env python3
"""
OpenAI Usage Monitor CLI

This script provides command-line tools to monitor OpenAI usage,
check quota status, and manage rate limiting.
"""

import sys
import os
import argparse
from datetime import datetime
import json

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.shared.openai_rate_limiter import get_openai_usage_stats, reset_openai_quota, check_quota_available
from app.shared.openai_config import get_openai_config, estimate_cost
from app.shared.redis_client import redis_client

def display_usage_stats(hours: int = 24):
    """Display OpenAI usage statistics."""
    print(f"\n🔍 OpenAI Usage Statistics (Last {hours} hours)")
    print("=" * 60)
    
    try:
        stats = get_openai_usage_stats(hours)
        
        if "error" in stats:
            print(f"❌ Error retrieving stats: {stats['error']}")
            return
        
        print(f"📊 Model: {stats['model']}")
        print(f"📈 Total Requests: {stats['total_requests']}")
        print(f"🎯 Total Tokens: {stats['total_tokens']:,}")
        
        # Estimate total cost
        if stats['total_tokens'] > 0:
            config = get_openai_config()
            estimated_cost = (stats['total_tokens'] / 1000) * config.cost_per_1k_tokens
            print(f"💰 Estimated Cost: ${estimated_cost:.4f}")
        
        print(f"\n📅 Hourly Breakdown:")
        print("-" * 40)
        
        for entry in stats['hourly_breakdown'][-12:]:  # Show last 12 hours
            dt = datetime.fromisoformat(entry['datetime'].replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:00')
            print(f"  {time_str}: {entry['requests']:>3} requests, {entry['tokens']:>6,} tokens")
        
        if len(stats['hourly_breakdown']) > 12:
            print(f"  ... and {len(stats['hourly_breakdown']) - 12} more hours")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def check_quota_status():
    """Check current quota status."""
    print("\n🔍 OpenAI Quota Status")
    print("=" * 30)
    
    try:
        import asyncio
        quota_available = asyncio.run(check_quota_available())
        
        if quota_available:
            print("✅ Quota Status: Available")
        else:
            print("❌ Quota Status: Exceeded")
            
            # Try to get more details from Redis
            quota_key = "openai:quota_status"
            quota_data = redis_client.get_json(quota_key)
            if quota_data:
                print(f"   Error: {quota_data.get('error', 'Unknown')}")
                reset_time = quota_data.get('reset_time')
                if reset_time:
                    reset_dt = datetime.fromisoformat(reset_time)
                    print(f"   Expected Reset: {reset_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    
    except Exception as e:
        print(f"❌ Error checking quota: {e}")

def display_config():
    """Display current OpenAI configuration."""
    print("\n⚙️  Current OpenAI Configuration")
    print("=" * 40)
    
    try:
        config = get_openai_config()
        
        print(f"🤖 Model: {config.model_name}")
        print(f"📏 Max Tokens: {config.max_tokens}")
        print(f"🌡️  Temperature: {config.temperature}")
        print(f"⏱️  Timeout: {config.timeout_sec}s")
        print(f"🔄 Max Retries: {config.max_retries}")
        print(f"💰 Cost per 1K tokens: ${config.cost_per_1k_tokens}")
        print(f"🚀 Rate Limit: {config.tokens_per_minute_limit:,} tokens/min")
        
    except Exception as e:
        print(f"❌ Error getting config: {e}")

def reset_quota():
    """Reset quota status."""
    print("\n🔄 Resetting OpenAI Quota Status")
    print("=" * 35)
    
    try:
        reset_openai_quota()
        print("✅ Quota status reset successfully")
        print("ℹ️  Note: This only resets the local tracking. If you've actually")
        print("   exceeded your OpenAI billing quota, you'll need to add credits.")
        
    except Exception as e:
        print(f"❌ Error resetting quota: {e}")

def estimate_request_cost(tokens: int):
    """Estimate cost for a given number of tokens."""
    print(f"\n💰 Cost Estimation for {tokens:,} tokens")
    print("=" * 40)
    
    try:
        config = get_openai_config()
        cost = estimate_cost(tokens // 2, tokens // 2)  # Assume equal prompt/completion
        
        print(f"🤖 Model: {config.model_name}")
        print(f"💵 Estimated Cost: ${cost:.6f}")
        print(f"📊 Cost per 1K tokens: ${config.cost_per_1k_tokens}")
        
        # Show scale examples
        print(f"\n📈 Cost Scale:")
        for scale_tokens in [1000, 10000, 100000, 1000000]:
            scale_cost = (scale_tokens / 1000) * config.cost_per_1k_tokens
            print(f"   {scale_tokens:>7,} tokens = ${scale_cost:.4f}")
            
    except Exception as e:
        print(f"❌ Error estimating cost: {e}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Monitor OpenAI usage and manage quota",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor_openai_usage.py stats --hours 48     # Show 48-hour usage stats
  python monitor_openai_usage.py quota                # Check quota status  
  python monitor_openai_usage.py config               # Show current config
  python monitor_openai_usage.py reset                # Reset quota status
  python monitor_openai_usage.py cost --tokens 5000   # Estimate cost for 5K tokens
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Usage stats command
    stats_parser = subparsers.add_parser('stats', help='Show usage statistics')
    stats_parser.add_argument('--hours', type=int, default=24,
                             help='Number of hours to analyze (default: 24)')
    
    # Quota status command
    subparsers.add_parser('quota', help='Check quota status')
    
    # Configuration command
    subparsers.add_parser('config', help='Show current configuration')
    
    # Reset quota command
    subparsers.add_parser('reset', help='Reset quota status')
    
    # Cost estimation command
    cost_parser = subparsers.add_parser('cost', help='Estimate request cost')
    cost_parser.add_argument('--tokens', type=int, required=True,
                            help='Number of tokens to estimate cost for')
    
    args = parser.parse_args()
    
    if args.command == 'stats':
        display_usage_stats(args.hours)
    elif args.command == 'quota':
        check_quota_status()
    elif args.command == 'config':
        display_config()
    elif args.command == 'reset':
        reset_quota()
    elif args.command == 'cost':
        estimate_request_cost(args.tokens)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
