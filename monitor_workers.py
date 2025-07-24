#!/usr/bin/env python3
"""
CLI tool to monitor Celery worker health and system status.
"""
import sys
import time
import json
from datetime import datetime
from app.workers.monitor import worker_monitor, WorkerStatus


def format_bytes(bytes_value):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def format_uptime(seconds):
    """Format uptime in human readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def print_worker_status():
    """Print detailed worker status."""
    print("=" * 80)
    print("CELERY WORKER STATUS")
    print("=" * 80)
    
    try:
        workers = worker_monitor.get_all_workers()
        
        if not workers:
            print("No workers found. Make sure workers are running.")
            return
        
        # Group workers by queue
        by_queue = {}
        for name, worker in workers.items():
            queue = worker.queue
            if queue not in by_queue:
                by_queue[queue] = []
            by_queue[queue].append((name, worker))
        
        for queue, queue_workers in by_queue.items():
            print(f"\n{queue.upper()} QUEUE:")
            print("-" * 40)
            
            for name, worker in queue_workers:
                status_color = {
                    WorkerStatus.HEALTHY: "🟢",
                    WorkerStatus.WARNING: "🟡", 
                    WorkerStatus.ERROR: "🔴",
                    WorkerStatus.OFFLINE: "⚫"
                }.get(worker.status, "❓")
                
                last_heartbeat = datetime.fromtimestamp(worker.last_heartbeat)
                time_ago = time.time() - worker.last_heartbeat
                
                print(f"{status_color} {name}")
                print(f"   PID: {worker.pid}")
                print(f"   Status: {worker.status.value}")
                print(f"   Memory: {format_bytes(worker.memory_usage)} ({worker.memory_percent:.1f}%)")
                print(f"   CPU: {worker.cpu_percent:.1f}%")
                print(f"   Uptime: {format_uptime(worker.uptime)}")
                print(f"   Last heartbeat: {last_heartbeat.strftime('%H:%M:%S')} ({time_ago:.0f}s ago)")
                
                if worker.warnings:
                    print(f"   ⚠️  Warnings:")
                    for warning in worker.warnings:
                        print(f"      - {warning}")
                print()
        
    except Exception as e:
        print(f"Error getting worker status: {e}")


def print_queue_stats():
    """Print queue statistics."""
    print("=" * 80)
    print("QUEUE STATISTICS")
    print("=" * 80)
    
    try:
        queue_stats = worker_monitor.get_queue_stats()
        
        for queue, stats in queue_stats.items():
            if "error" in stats:
                print(f"{queue}: Error - {stats['error']}")
                continue
                
            print(f"\n{queue.upper()}:")
            print(f"  Queue length: {stats['queue_length']}")
            print(f"  Active tasks: {stats['active_tasks']}")
            
            if stats['active_task_details']:
                print("  Active task details:")
                for task in stats['active_task_details']:
                    start_time = task.get('time_start', 'unknown')
                    print(f"    - {task['task_name']} [{task['task_id'][:8]}...] on {task['worker']}")
                    if start_time != 'unknown':
                        print(f"      Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"Error getting queue stats: {e}")


def print_system_health():
    """Print overall system health."""
    print("=" * 80)
    print("SYSTEM HEALTH")
    print("=" * 80)
    
    try:
        health = worker_monitor.get_system_health()
        
        status_emoji = {
            "healthy": "🟢",
            "warning": "🟡",
            "degraded": "🟠",
            "error": "🔴"
        }.get(health["system_status"], "❓")
        
        print(f"Overall Status: {status_emoji} {health['system_status'].upper()}")
        print(f"Timestamp: {datetime.fromtimestamp(health['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nWorkers: {health['workers']['total']} total")
        for status, count in health['workers']['by_status'].items():
            if count > 0:
                print(f"  {status}: {count}")
        
        print(f"\nQueues: {health['queues']['total_length']} total pending tasks")
        for queue, stats in health['queues']['by_queue'].items():
            if stats.get('queue_length', 0) > 0:
                print(f"  {queue}: {stats['queue_length']} pending, {stats['active_tasks']} active")
        
        print(f"\nRedis: {'🟢 Connected' if health['redis_health'] else '🔴 Disconnected'}")
        print(f"Database: {'🟢 Accessible' if health['database_accessible'] else '🔴 Inaccessible'}")
        
    except Exception as e:
        print(f"Error getting system health: {e}")


def main():
    """Main CLI function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "workers":
            print_worker_status()
        elif command == "queues":
            print_queue_stats()
        elif command == "health":
            print_system_health()
        elif command == "all":
            print_system_health()
            print_worker_status()
            print_queue_stats()
        elif command == "watch":
            # Watch mode - refresh every 5 seconds
            try:
                while True:
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_system_health()
                    print_worker_status()
                    print("\nPress Ctrl+C to exit watch mode...")
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\nExiting watch mode.")
        elif command == "cleanup":
            result = worker_monitor.cleanup_stale_worker_data()
            print(f"Cleaned up {len(result['cleaned_workers'])} stale workers")
            if result['cleaned_workers']:
                print("Cleaned workers:", result['cleaned_workers'])
        else:
            print(f"Unknown command: {command}")
            print_help()
    else:
        print_system_health()


def print_help():
    """Print help message."""
    print("Usage: python monitor_workers.py [command]")
    print("\nCommands:")
    print("  workers  - Show detailed worker status")
    print("  queues   - Show queue statistics")
    print("  health   - Show system health (default)")
    print("  all      - Show all information")
    print("  watch    - Watch mode (refresh every 5 seconds)")
    print("  cleanup  - Clean up stale worker data")
    print("  help     - Show this help message")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print_help()
    else:
        main()