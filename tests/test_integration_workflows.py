"""
Integration tests for complete user workflows.
Tests end-to-end functionality across multiple components.
"""
import pytest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.main import app
from app.shared.models import Job, JobStatus, JobType, UserSession
from app.shared.session_manager import SessionManager
from app.shared.auth import AuthenticationManager
from app.shared.job_manager import JobManager
from app.shared.resource_manager import ResourceManager


class TestCompleteUserWorkflows:
    """Test complete user workflows from authentication to results."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch('app.shared.auth.config') as mock_config, \
             patch('app.shared.session_manager.redis_client') as mock_redis, \
             patch('app.shared.job_manager.redis_client') as mock_job_redis, \
             patch('app.shared.resource_manager.redis_client') as mock_resource_redis:
            
            # Configure auth
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                "test_user": {"password": "test_pass", "groups": ["test_group"]}
            }
            
            # Configure Redis mocks
            mock_redis.set_session.return_value = True
            mock_redis.get_session.return_value = None
            mock_redis.delete_session.return_value = True
            
            mock_job_redis.get_user_jobs.return_value = []
            mock_job_redis.set_job.return_value = True
            mock_job_redis.get_job.return_value = None
            
            mock_resource_redis.client.keys.return_value = []
            mock_resource_redis.get_json.return_value = None
            mock_resource_redis.set_json.return_value = True
            
            yield {
                "config": mock_config,
                "redis": mock_redis,
                "job_redis": mock_job_redis,
                "resource_redis": mock_resource_redis
            }
    
    def test_complete_document_upload_workflow(self, client, mock_dependencies):
        """Test complete document upload workflow from auth to completion."""
        # Step 1: Authenticate user
        with patch('app.shared.auth.session_manager') as mock_session_manager:
            mock_session = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            
            # Authenticate
            auth_data = {"username": "test_user", "password": "test_pass"}
            auth_response = client.post("/api/auth/login", json=auth_data)
            
            assert auth_response.status_code == 200
            token = auth_response.json()["access_token"]
            
            # Step 2: Upload document
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                upload_job = Job(
                    user_id="test_user",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.PENDING,
                    metadata={"filename": "test_document.pdf"}
                )
                mock_job_manager.create_job.return_value = upload_job
                mock_job_manager.get_job.return_value = upload_job
                
                # Upload file
                headers = {"Authorization": f"Bearer {token}"}
                files = {"file": ("test_document.pdf", b"PDF content", "application/pdf")}
                upload_response = client.post("/api/documents/upload", files=files, headers=headers)
                
                assert upload_response.status_code == 200
                job_id = upload_response.json()["job_id"]
                
                # Step 3: Monitor upload progress
                upload_job.status = JobStatus.PROCESSING
                upload_job.progress = 0.5
                
                status_response = client.get(f"/api/jobs/{job_id}", headers=headers)
                assert status_response.status_code == 200
                assert status_response.json()["status"] == "processing"
                assert status_response.json()["progress"] == 0.5
                
                # Step 4: Complete upload
                upload_job.status = JobStatus.COMPLETED
                upload_job.progress = 1.0
                upload_job.result = {"pages_processed": 10, "chunks_created": 50}
                
                final_status_response = client.get(f"/api/jobs/{job_id}", headers=headers)
                assert final_status_response.status_code == 200
                assert final_status_response.json()["status"] == "completed"
                assert final_status_response.json()["result"]["pages_processed"] == 10
    
    def test_complete_query_workflow(self, client, mock_dependencies):
        """Test complete query workflow from submission to results."""
        # Setup authenticated user
        with patch('app.shared.auth.session_manager') as mock_session_manager:
            mock_session = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            
            # Authenticate
            auth_data = {"username": "test_user", "password": "test_pass"}
            auth_response = client.post("/api/auth/login", json=auth_data)
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Step 1: Submit query
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                query_job = Job(
                    user_id="test_user",
                    job_type=JobType.QUERY,
                    status=JobStatus.PENDING,
                    metadata={"query_text": "What is artificial intelligence?"}
                )
                mock_job_manager.create_job.return_value = query_job
                mock_job_manager.get_job.return_value = query_job
                
                query_data = {"query": "What is artificial intelligence?"}
                query_response = client.post("/api/query", json=query_data, headers=headers)
                
                assert query_response.status_code == 200
                query_id = query_response.json()["query_id"]
                
                # Step 2: Monitor query processing
                query_job.status = JobStatus.PROCESSING
                query_job.progress = 0.3
                
                status_response = client.get(f"/api/query/{query_id}/status", headers=headers)
                assert status_response.status_code == 200
                assert status_response.json()["status"] == "processing"
                
                # Step 3: Get completed results
                query_job.status = JobStatus.COMPLETED
                query_job.result = {
                    "answer": "Artificial intelligence is the simulation of human intelligence...",
                    "sources": [
                        {"document": "ai_basics.pdf", "page": 1, "relevance": 0.95},
                        {"document": "ml_guide.pdf", "page": 3, "relevance": 0.87}
                    ],
                    "processing_time": 2.3
                }
                
                result_response = client.get(f"/api/query/{query_id}", headers=headers)
                assert result_response.status_code == 200
                result_data = result_response.json()
                
                assert result_data["status"] == "completed"
                assert "answer" in result_data["result"]
                assert len(result_data["result"]["sources"]) == 2
                assert result_data["result"]["processing_time"] == 2.3
    
    def test_multi_user_concurrent_workflow(self, client, mock_dependencies):
        """Test concurrent workflows from multiple users."""
        workflow_results = []
        errors = []
        
        def user_workflow(user_id):
            try:
                # Setup user-specific mocks
                with patch('app.shared.auth.session_manager') as mock_session_manager:
                    mock_session = UserSession(
                        session_id=f"session_{user_id}",
                        user_id=user_id,
                        groups=[f"group_{user_id}"],
                        permissions=["upload", "query"]
                    )
                    mock_session_manager.create_session.return_value = mock_session
                    mock_session_manager.validate_session.return_value = True
                    mock_session_manager.update_session_activity.return_value = True
                    mock_session_manager.get_session.return_value = mock_session
                    
                    # Update config for this user
                    mock_dependencies["config"].USERS[user_id] = {
                        "password": f"pass_{user_id}",
                        "groups": [f"group_{user_id}"]
                    }
                    
                    # Step 1: Authenticate
                    auth_data = {"username": user_id, "password": f"pass_{user_id}"}
                    auth_response = client.post("/api/auth/login", json=auth_data)
                    
                    if auth_response.status_code != 200:
                        raise Exception(f"Auth failed for {user_id}")
                    
                    token = auth_response.json()["access_token"]
                    headers = {"Authorization": f"Bearer {token}"}
                    
                    # Step 2: Upload document
                    with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                        upload_job = Job(
                            user_id=user_id,
                            job_type=JobType.EMBEDDING,
                            status=JobStatus.COMPLETED,
                            metadata={"filename": f"{user_id}_document.pdf"}
                        )
                        mock_job_manager.create_job.return_value = upload_job
                        mock_job_manager.get_job.return_value = upload_job
                        
                        files = {"file": (f"{user_id}_document.pdf", b"PDF content", "application/pdf")}
                        upload_response = client.post("/api/documents/upload", files=files, headers=headers)
                        
                        if upload_response.status_code != 200:
                            raise Exception(f"Upload failed for {user_id}")
                        
                        # Step 3: Submit query
                        query_job = Job(
                            user_id=user_id,
                            job_type=JobType.QUERY,
                            status=JobStatus.COMPLETED,
                            metadata={"query_text": f"Query from {user_id}"},
                            result={"answer": f"Answer for {user_id}"}
                        )
                        mock_job_manager.create_job.return_value = query_job
                        
                        query_data = {"query": f"Query from {user_id}"}
                        query_response = client.post("/api/query", json=query_data, headers=headers)
                        
                        if query_response.status_code != 200:
                            raise Exception(f"Query failed for {user_id}")
                        
                        workflow_results.append({
                            "user_id": user_id,
                            "upload_job_id": upload_response.json()["job_id"],
                            "query_id": query_response.json()["query_id"],
                            "success": True
                        })
                        
            except Exception as e:
                errors.append(f"User {user_id}: {e}")
                workflow_results.append({
                    "user_id": user_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Run concurrent workflows
        threads = []
        for i in range(5):
            user_id = f"user_{i}"
            thread = threading.Thread(target=user_workflow, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(workflow_results) == 5
        
        successful_workflows = [r for r in workflow_results if r["success"]]
        assert len(successful_workflows) == 5
        
        # Verify unique job IDs
        upload_job_ids = [r["upload_job_id"] for r in successful_workflows]
        query_ids = [r["query_id"] for r in successful_workflows]
        
        assert len(set(upload_job_ids)) == 5
        assert len(set(query_ids)) == 5
    
    def test_error_recovery_workflow(self, client, mock_dependencies):
        """Test error recovery in workflows."""
        with patch('app.shared.auth.session_manager') as mock_session_manager:
            mock_session = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            
            # Authenticate
            auth_data = {"username": "test_user", "password": "test_pass"}
            auth_response = client.post("/api/auth/login", json=auth_data)
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test failed job recovery
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                # Create a failed job
                failed_job = Job(
                    user_id="test_user",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.FAILED,
                    metadata={"filename": "failed_document.pdf"},
                    error="Processing failed due to corrupted file"
                )
                mock_job_manager.create_job.return_value = failed_job
                mock_job_manager.get_job.return_value = failed_job
                
                # Upload file that will fail
                files = {"file": ("failed_document.pdf", b"corrupted content", "application/pdf")}
                upload_response = client.post("/api/documents/upload", files=files, headers=headers)
                
                assert upload_response.status_code == 200
                job_id = upload_response.json()["job_id"]
                
                # Check failed status
                status_response = client.get(f"/api/jobs/{job_id}", headers=headers)
                assert status_response.status_code == 200
                assert status_response.json()["status"] == "failed"
                assert "Processing failed" in status_response.json()["error"]
                
                # Test retry mechanism
                retry_job = Job(
                    user_id="test_user",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.COMPLETED,
                    metadata={"filename": "retry_document.pdf"}
                )
                mock_job_manager.create_job.return_value = retry_job
                
                # Retry with corrected file
                retry_files = {"file": ("retry_document.pdf", b"valid PDF content", "application/pdf")}
                retry_response = client.post("/api/documents/upload", files=retry_files, headers=headers)
                
                assert retry_response.status_code == 200
                assert retry_response.json()["status"] == "queued"


class TestConcurrentUserTesting:
    """Test system behavior under concurrent user load."""
    
    def test_concurrent_authentication_load(self):
        """Test authentication system under concurrent load."""
        auth_manager = AuthenticationManager()
        results = []
        errors = []
        
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                f"user_{i}": {"password": f"pass_{i}", "groups": [f"group_{i}"]}
                for i in range(50)  # 50 concurrent users
            }
            
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                def create_session_side_effect(user_id, groups, permissions):
                    return UserSession(
                        session_id=f"session_{user_id}_{time.time()}",
                        user_id=user_id,
                        groups=groups,
                        permissions=permissions
                    )
                
                mock_session_manager.create_session.side_effect = create_session_side_effect
                mock_session_manager.validate_session.return_value = True
                mock_session_manager.update_session_activity.return_value = True
                
                def auth_worker(user_num):
                    try:
                        start_time = time.time()
                        
                        user_id = f"user_{user_num}"
                        password = f"pass_{user_num}"
                        
                        # Authenticate
                        auth_result = auth_manager.authenticate_user(user_id, password)
                        
                        # Validate token
                        user_info = auth_manager.validate_token(auth_result["access_token"])
                        
                        # Check permissions
                        has_query = auth_manager.has_permission(auth_result["access_token"], "query")
                        
                        end_time = time.time()
                        
                        results.append({
                            "user_id": user_id,
                            "auth_time": end_time - start_time,
                            "token_valid": user_info is not None,
                            "has_permissions": has_query,
                            "success": True
                        })
                        
                    except Exception as e:
                        errors.append(f"User {user_num}: {e}")
                        results.append({
                            "user_id": f"user_{user_num}",
                            "success": False,
                            "error": str(e)
                        })
                
                # Run concurrent authentication
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = [executor.submit(auth_worker, i) for i in range(50)]
                    
                    for future in as_completed(futures):
                        future.result()
                
                total_time = time.time() - start_time
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50
        
        successful_auths = [r for r in results if r["success"]]
        assert len(successful_auths) == 50
        
        # Performance checks
        avg_auth_time = sum(r["auth_time"] for r in successful_auths) / len(successful_auths)
        assert avg_auth_time < 0.1  # Average auth time should be under 100ms
        assert total_time < 10  # Total time should be under 10 seconds
        
        # Verify all tokens are valid and have permissions
        for result in successful_auths:
            assert result["token_valid"] is True
            assert result["has_permissions"] is True
    
    def test_concurrent_job_processing_load(self):
        """Test job processing system under concurrent load."""
        job_manager = JobManager()
        resource_manager = ResourceManager()
        
        job_results = []
        errors = []
        
        with patch('app.shared.job_manager.redis_client') as mock_redis, \
             patch('app.shared.resource_manager.redis_client') as mock_resource_redis:
            
            mock_redis.get_user_jobs.return_value = []
            mock_redis.set_job.return_value = True
            mock_redis.get_job.return_value = None
            
            mock_resource_redis.client.keys.return_value = []
            mock_resource_redis.get_json.return_value = None
            mock_resource_redis.set_json.return_value = True
            
            def job_worker(user_id, job_num):
                try:
                    start_time = time.time()
                    
                    # Check resource availability
                    can_accept = resource_manager.can_accept_task("embedding")
                    
                    if can_accept:
                        # Create job
                        job = job_manager.create_job(
                            user_id=user_id,
                            job_type=JobType.EMBEDDING,
                            metadata={"document": f"doc_{job_num}.pdf"}
                        )
                        
                        # Simulate job processing
                        job_manager.update_job_status(job.job_id, JobStatus.PROCESSING)
                        time.sleep(0.01)  # Simulate work
                        job_manager.update_job_status(job.job_id, JobStatus.COMPLETED)
                        
                        end_time = time.time()
                        
                        job_results.append({
                            "user_id": user_id,
                            "job_id": job.job_id,
                            "processing_time": end_time - start_time,
                            "success": True
                        })
                    else:
                        job_results.append({
                            "user_id": user_id,
                            "success": False,
                            "reason": "Resource unavailable"
                        })
                        
                except Exception as e:
                    errors.append(f"User {user_id}, Job {job_num}: {e}")
                    job_results.append({
                        "user_id": user_id,
                        "success": False,
                        "error": str(e)
                    })
            
            # Create jobs from multiple users
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for user_num in range(10):
                    for job_num in range(5):
                        user_id = f"user_{user_num}"
                        future = executor.submit(job_worker, user_id, job_num)
                        futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
            
            total_time = time.time() - start_time
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(job_results) == 50  # 10 users * 5 jobs each
        
        successful_jobs = [r for r in job_results if r["success"]]
        
        # At least some jobs should succeed (depending on resource limits)
        assert len(successful_jobs) > 0
        
        # Performance checks for successful jobs
        if successful_jobs:
            avg_processing_time = sum(r["processing_time"] for r in successful_jobs) / len(successful_jobs)
            assert avg_processing_time < 0.5  # Average processing time under 500ms
        
        # Verify job distribution across users
        user_job_counts = {}
        for result in successful_jobs:
            user_id = result["user_id"]
            user_job_counts[user_id] = user_job_counts.get(user_id, 0) + 1
        
        # Jobs should be distributed across multiple users
        assert len(user_job_counts) > 1
    
    def test_websocket_concurrent_connections(self):
        """Test WebSocket system under concurrent connections."""
        from app.shared.websocket_manager import WebSocketManager
        
        websocket_manager = WebSocketManager()
        connection_results = []
        
        async def simulate_websocket_connection(user_id):
            try:
                # Simulate WebSocket connection
                mock_websocket = Mock()
                mock_websocket.send_text = AsyncMock()
                mock_websocket.close = AsyncMock()
                
                # Connect user
                await websocket_manager.connect_user(user_id, mock_websocket)
                
                # Send test message
                await websocket_manager.send_to_user(user_id, {
                    "type": "test_message",
                    "data": f"Message for {user_id}"
                })
                
                # Simulate some activity
                await asyncio.sleep(0.01)
                
                # Disconnect user
                await websocket_manager.disconnect_user(user_id)
                
                connection_results.append({
                    "user_id": user_id,
                    "success": True
                })
                
            except Exception as e:
                connection_results.append({
                    "user_id": user_id,
                    "success": False,
                    "error": str(e)
                })
        
        async def run_concurrent_connections():
            # Create concurrent WebSocket connections
            tasks = []
            for i in range(20):
                user_id = f"ws_user_{i}"
                task = asyncio.create_task(simulate_websocket_connection(user_id))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Run the test
        asyncio.run(run_concurrent_connections())
        
        # Verify results
        assert len(connection_results) == 20
        
        successful_connections = [r for r in connection_results if r["success"]]
        assert len(successful_connections) == 20
        
        # Verify unique user connections
        user_ids = [r["user_id"] for r in successful_connections]
        assert len(set(user_ids)) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])