#!/usr/bin/env python3
"""
Job cleanup utility to fix stuck jobs and resolve concurrent job limit issues.

This script helps diagnose and fix situations where users are hitting the 
maximum concurrent jobs limit due to stuck or orphaned jobs.
"""
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.shared.job_manager import job_manager
from app.shared.redis_client import redis_client
from app.shared.models import JobStatus
from datetime import datetime, timedelta
import argparse

def diagnose_user_jobs(user_id: str):
    """Diagnose job issues for a specific user."""
    print(f"\n=== Diagnosing jobs for user: {user_id} ===")
    
    # Get all jobs for the user
    all_jobs = job_manager.get_user_jobs(user_id, limit=50)
    active_jobs = job_manager.get_user_jobs(user_id, active_only=True)
    
    print(f"Total jobs: {len(all_jobs)}")
    print(f"Active jobs: {len(active_jobs)}")
    
    # Analyze job status distribution
    status_counts = {}
    for job in all_jobs:
        status = job.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nJob Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Check for stuck jobs (processing for too long)
    now = datetime.now()
    stuck_jobs = []
    
    for job in active_jobs:
        if job.status == JobStatus.PROCESSING and job.started_at:
            duration = now - job.started_at
            if duration > timedelta(hours=2):  # Jobs running for more than 2 hours
                stuck_jobs.append(job)
    
    if stuck_jobs:
        print(f"\nFound {len(stuck_jobs)} potentially stuck jobs:")
        for job in stuck_jobs:
            duration = now - job.started_at
            print(f"  Job {job.job_id}: {job.job_type.value}, running for {duration}")
    
    # Check for very old pending jobs
    old_pending = []
    for job in all_jobs:
        if job.status == JobStatus.PENDING:
            age = now - job.created_at
            if age > timedelta(hours=1):  # Pending for more than 1 hour
                old_pending.append(job)
    
    if old_pending:
        print(f"\nFound {len(old_pending)} old pending jobs:")
        for job in old_pending:
            age = now - job.created_at
            print(f"  Job {job.job_id}: {job.job_type.value}, pending for {age}")
    
    return {
        'all_jobs': all_jobs,
        'active_jobs': active_jobs,
        'stuck_jobs': stuck_jobs,
        'old_pending': old_pending,
        'status_counts': status_counts
    }

def cleanup_stuck_jobs(user_id: str, dry_run: bool = True):
    """Clean up stuck jobs for a user."""
    print(f"\n=== Cleaning up stuck jobs for user: {user_id} ===")
    
    diagnosis = diagnose_user_jobs(user_id)
    stuck_jobs = diagnosis['stuck_jobs']
    old_pending = diagnosis['old_pending']
    
    actions_taken = 0
    
    # Cancel stuck processing jobs
    for job in stuck_jobs:
        action = "Would cancel" if dry_run else "Cancelling"
        print(f"{action} stuck job {job.job_id} (running for {datetime.now() - job.started_at})")
        
        if not dry_run:
            success = job_manager.cancel_job(job.job_id, "Cancelled by cleanup script - stuck job")
            if success:
                actions_taken += 1
                print(f"  ✓ Successfully cancelled job {job.job_id}")
            else:
                print(f"  ✗ Failed to cancel job {job.job_id}")
    
    # Cancel old pending jobs
    for job in old_pending:
        action = "Would cancel" if dry_run else "Cancelling"
        print(f"{action} old pending job {job.job_id} (pending for {datetime.now() - job.created_at})")
        
        if not dry_run:
            success = job_manager.cancel_job(job.job_id, "Cancelled by cleanup script - old pending job")
            if success:
                actions_taken += 1
                print(f"  ✓ Successfully cancelled job {job.job_id}")
            else:
                print(f"  ✗ Failed to cancel job {job.job_id}")
    
    if dry_run:
        total_would_clean = len(stuck_jobs) + len(old_pending)
        print(f"\nDry run complete. Would clean up {total_would_clean} jobs.")
        print("Run with --execute to actually perform cleanup.")
    else:
        print(f"\nCleanup complete. Cleaned up {actions_taken} jobs.")
    
    return actions_taken

def cleanup_all_stuck_jobs(dry_run: bool = True):
    """Clean up stuck jobs for all users."""
    print("\n=== Cleaning up stuck jobs for all users ===")
    
    # Get all jobs that might be stuck
    now = datetime.now()
    
    # Get processing jobs that are too old
    processing_jobs = job_manager.get_jobs_by_status(JobStatus.PROCESSING)
    stuck_processing = []
    
    for job in processing_jobs:
        if job.started_at:
            duration = now - job.started_at
            if duration > timedelta(hours=2):  # Jobs running for more than 2 hours
                stuck_processing.append(job)
    
    # Get pending jobs that are too old
    pending_jobs = job_manager.get_jobs_by_status(JobStatus.PENDING)
    old_pending = []
    
    for job in pending_jobs:
        age = now - job.created_at
        if age > timedelta(hours=1):  # Pending for more than 1 hour
            old_pending.append(job)
    
    print(f"Found {len(stuck_processing)} stuck processing jobs")
    print(f"Found {len(old_pending)} old pending jobs")
    
    actions_taken = 0
    
    # Cancel stuck processing jobs
    for job in stuck_processing:
        action = "Would cancel" if dry_run else "Cancelling"
        duration = now - job.started_at
        print(f"{action} stuck job {job.job_id} for user {job.user_id} (running for {duration})")
        
        if not dry_run:
            success = job_manager.cancel_job(job.job_id, "Cancelled by cleanup script - stuck job")
            if success:
                actions_taken += 1
    
    # Cancel old pending jobs
    for job in old_pending:
        action = "Would cancel" if dry_run else "Cancelling"
        age = now - job.created_at
        print(f"{action} old pending job {job.job_id} for user {job.user_id} (pending for {age})")
        
        if not dry_run:
            success = job_manager.cancel_job(job.job_id, "Cancelled by cleanup script - old pending job")
            if success:
                actions_taken += 1
    
    if dry_run:
        total_would_clean = len(stuck_processing) + len(old_pending)
        print(f"\nDry run complete. Would clean up {total_would_clean} jobs.")
    else:
        print(f"\nCleanup complete. Cleaned up {actions_taken} jobs.")
    
    return actions_taken

def get_system_stats():
    """Get overall system statistics."""
    print("\n=== System Statistics ===")
    stats = job_manager.get_job_statistics()
    
    print(f"Total jobs: {stats.get('total_jobs', 0)}")
    print(f"Active jobs: {stats.get('active_jobs', 0)}")
    print(f"Completed jobs: {stats.get('completed_jobs', 0)}")
    print(f"Failed jobs: {stats.get('failed_jobs', 0)}")
    print(f"Success rate: {stats.get('success_rate', 0):.2%}")
    print(f"Average duration: {stats.get('average_duration', 0):.2f}s")
    
    print("\nJob Status Distribution:")
    for status, count in stats.get('by_status', {}).items():
        if count > 0:
            print(f"  {status}: {count}")
    
    print("\nJob Type Distribution:")
    for job_type, count in stats.get('by_type', {}).items():
        if count > 0:
            print(f"  {job_type}: {count}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Job cleanup utility")
    parser.add_argument("--user", "-u", help="Target specific user ID")
    parser.add_argument("--execute", "-x", action="store_true", help="Execute cleanup (default is dry run)")
    parser.add_argument("--stats", "-s", action="store_true", help="Show system statistics")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Diagnose issues without cleanup")
    
    args = parser.parse_args()
    
    try:
        # Test Redis connection
        if not redis_client.health_check():
            print("❌ Redis connection failed. Make sure Redis is running.")
            return 1
        
        print("✅ Redis connection successful")
        
        if args.stats:
            get_system_stats()
            return 0
        
        if args.user:
            if args.diagnose:
                diagnose_user_jobs(args.user)
            else:
                cleanup_stuck_jobs(args.user, dry_run=not args.execute)
        else:
            if args.diagnose:
                get_system_stats()
            else:
                cleanup_all_stuck_jobs(dry_run=not args.execute)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
