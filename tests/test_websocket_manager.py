"""
Tests for WebSocket connection management.
"""
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import WebSocket
from fastapi.testclient import TestClient

from app.shared.websocket_manager import (
    WebSocketManager, 
    WebSocketConnection, 
    WebSocketAuthenticationError,
    WebSocketConnectionError,
    websocket_manager
)
from app.shared.models import UserSession
from app.shared.auth import TokenInvalidError, TokenExpiredError


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.closed = False
        self.close_code = None
        self.accepted = False
    
    async def accept(self):
        """Mock accept method."""
        self.accepted = True
    
    async def send_text(self, message: str):
        """Mock send_text method."""
        if self.closed:
            raise Exception("WebSocket is closed")
        self.messages_sent.append(message)
    
    async def receive_text(self) -> str:
        """Mock receive_text method."""
        if self.closed:
            raise Exception("WebSocket is closed")
        if not self.messages_received:
            raise Exception("No messages to receive")
        return self.messages_received.pop(0)
    
    async def close(self, code: int = 1000):
        """Mock close method."""
        self.closed = True
        self.close_code = code
    
    def add_message(self, message: str):
        """Add message to receive queue."""
        self.messages_received.append(message)


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket."""
    return MockWebSocket()


@pytest.fixture
def test_user_session():
    """Create test user session."""
    return UserSession(
        session_id="test-session-123",
        user_id="test-user",
        groups=["group1", "group2"],
        permissions=["upload", "query"],
        created_at=datetime.now(),
        last_activity=datetime.now(),
        is_active=True
    )


@pytest.fixture
def websocket_manager_instance():
    """Create fresh WebSocket manager instance for testing."""
    return WebSocketManager()


class TestWebSocketConnection:
    """Test WebSocketConnection class."""
    
    def test_connection_initialization(self, mock_websocket, test_user_session):
        """Test WebSocket connection initialization."""
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        assert connection.websocket == mock_websocket
        assert connection.user_session == test_user_session
        assert connection.connection_id is not None
        assert len(connection.connection_id) > 0
        assert connection.subscriptions == set()
        assert isinstance(connection.connected_at, datetime)
        assert isinstance(connection.last_ping, datetime)
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_websocket, test_user_session):
        """Test successful message sending."""
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        message = {"type": "test", "data": "hello"}
        result = await connection.send_message(message)
        
        assert result is True
        assert len(mock_websocket.messages_sent) == 1
        
        sent_message = json.loads(mock_websocket.messages_sent[0])
        assert sent_message["type"] == "test"
        assert sent_message["data"] == "hello"
        assert "timestamp" in sent_message
        assert sent_message["connection_id"] == connection.connection_id
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, test_user_session):
        """Test message sending failure."""
        # Create a WebSocket that raises exception on send
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Send failed"))
        
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        message = {"type": "test", "data": "hello"}
        result = await connection.send_message(message)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_error(self, mock_websocket, test_user_session):
        """Test error message sending."""
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        await connection.send_error("TEST_ERROR", "Test error message")
        
        assert len(mock_websocket.messages_sent) == 1
        sent_message = json.loads(mock_websocket.messages_sent[0])
        
        assert sent_message["type"] == "error"
        assert sent_message["error"]["code"] == "TEST_ERROR"
        assert sent_message["error"]["message"] == "Test error message"
    
    def test_topic_subscription(self, mock_websocket, test_user_session):
        """Test topic subscription management."""
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        # Test subscription
        connection.subscribe_to_topic("test-topic")
        assert "test-topic" in connection.subscriptions
        assert connection.is_subscribed_to("test-topic")
        
        # Test unsubscription
        connection.unsubscribe_from_topic("test-topic")
        assert "test-topic" not in connection.subscriptions
        assert not connection.is_subscribed_to("test-topic")
    
    def test_ping_update(self, mock_websocket, test_user_session):
        """Test ping timestamp update."""
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        original_ping = connection.last_ping
        connection.update_ping()
        
        assert connection.last_ping > original_ping


class TestWebSocketManager:
    """Test WebSocketManager class."""
    
    @pytest.mark.asyncio
    async def test_authenticate_connection_success(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test successful WebSocket authentication."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            result = await websocket_manager_instance.authenticate_connection(mock_websocket, "valid-token")
            
            assert result == test_user_session
            mock_auth.get_current_user.assert_called_once_with("valid-token")
    
    @pytest.mark.asyncio
    async def test_authenticate_connection_invalid_token(self, websocket_manager_instance, mock_websocket):
        """Test WebSocket authentication with invalid token."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.side_effect = TokenInvalidError("Invalid token")
            
            with pytest.raises(WebSocketAuthenticationError):
                await websocket_manager_instance.authenticate_connection(mock_websocket, "invalid-token")
    
    @pytest.mark.asyncio
    async def test_authenticate_connection_expired_token(self, websocket_manager_instance, mock_websocket):
        """Test WebSocket authentication with expired token."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.side_effect = TokenExpiredError("Token expired")
            
            with pytest.raises(WebSocketAuthenticationError):
                await websocket_manager_instance.authenticate_connection(mock_websocket, "expired-token")
    
    @pytest.mark.asyncio
    async def test_authenticate_connection_inactive_session(self, websocket_manager_instance, mock_websocket):
        """Test WebSocket authentication with inactive session."""
        inactive_session = UserSession(
            session_id="inactive-session",
            user_id="test-user",
            groups=["group1"],
            permissions=["query"],
            is_active=False
        )
        
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = inactive_session
            
            with pytest.raises(WebSocketAuthenticationError):
                await websocket_manager_instance.authenticate_connection(mock_websocket, "token")
    
    @pytest.mark.asyncio
    async def test_connect_success(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test successful WebSocket connection."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            
            assert mock_websocket.accepted
            assert connection.user_session == test_user_session
            assert connection.connection_id in websocket_manager_instance.connections
            assert test_user_session.user_id in websocket_manager_instance.user_connections
            
            # Check auto-subscriptions
            user_topic = f"user:{test_user_session.user_id}"
            assert connection.is_subscribed_to(user_topic)
            
            for group in test_user_session.groups:
                group_topic = f"group:{group}"
                assert connection.is_subscribed_to(group_topic)
            
            # Check welcome message was sent
            assert len(mock_websocket.messages_sent) == 1
            welcome_message = json.loads(mock_websocket.messages_sent[0])
            assert welcome_message["type"] == "connection_established"
    
    @pytest.mark.asyncio
    async def test_connect_authentication_failure(self, websocket_manager_instance, mock_websocket):
        """Test WebSocket connection with authentication failure."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.side_effect = TokenInvalidError("Invalid token")
            
            with pytest.raises(WebSocketAuthenticationError):
                await websocket_manager_instance.connect(mock_websocket, "invalid-token")
            
            # Check error message was sent and connection was closed
            assert len(mock_websocket.messages_sent) == 1
            error_message = json.loads(mock_websocket.messages_sent[0])
            assert error_message["type"] == "error"
            assert error_message["error"]["code"] == "AUTHENTICATION_FAILED"
            assert mock_websocket.closed
            assert mock_websocket.close_code == 4001
    
    @pytest.mark.asyncio
    async def test_disconnect(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test WebSocket disconnection and cleanup."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            # Connect first
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            connection_id = connection.connection_id
            
            # Verify connection exists
            assert connection_id in websocket_manager_instance.connections
            assert test_user_session.user_id in websocket_manager_instance.user_connections
            
            # Disconnect
            await websocket_manager_instance.disconnect(connection_id)
            
            # Verify cleanup
            assert connection_id not in websocket_manager_instance.connections
            assert test_user_session.user_id not in websocket_manager_instance.user_connections
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test handling ping message."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            
            # Clear welcome message
            mock_websocket.messages_sent.clear()
            
            # Handle ping message
            ping_message = json.dumps({"type": "ping"})
            await websocket_manager_instance.handle_message(connection.connection_id, ping_message)
            
            # Check pong response
            assert len(mock_websocket.messages_sent) == 1
            pong_message = json.loads(mock_websocket.messages_sent[0])
            assert pong_message["type"] == "pong"
            assert pong_message["message"] == "pong"
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test handling topic subscription message."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            mock_websocket.messages_sent.clear()
            
            # Subscribe to valid topic
            subscribe_message = json.dumps({"type": "subscribe", "topic": "public:announcements"})
            await websocket_manager_instance.handle_message(connection.connection_id, subscribe_message)
            
            # Check subscription confirmation
            assert len(mock_websocket.messages_sent) == 1
            confirmation = json.loads(mock_websocket.messages_sent[0])
            assert confirmation["type"] == "subscription_confirmed"
            assert confirmation["topic"] == "public:announcements"
            
            # Check connection is subscribed
            assert connection.is_subscribed_to("public:announcements")
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_invalid_topic(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test handling subscription to invalid topic."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            mock_websocket.messages_sent.clear()
            
            # Try to subscribe to unauthorized topic
            subscribe_message = json.dumps({"type": "subscribe", "topic": "admin:secrets"})
            await websocket_manager_instance.handle_message(connection.connection_id, subscribe_message)
            
            # Check error response
            assert len(mock_websocket.messages_sent) == 1
            error_message = json.loads(mock_websocket.messages_sent[0])
            assert error_message["type"] == "error"
            assert error_message["error"]["code"] == "INVALID_TOPIC"
    
    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test handling topic unsubscription message."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            
            # Subscribe first
            connection.subscribe_to_topic("public:test")
            websocket_manager_instance._add_topic_subscription("public:test", connection.connection_id)
            
            mock_websocket.messages_sent.clear()
            
            # Unsubscribe
            unsubscribe_message = json.dumps({"type": "unsubscribe", "topic": "public:test"})
            await websocket_manager_instance.handle_message(connection.connection_id, unsubscribe_message)
            
            # Check unsubscription confirmation
            assert len(mock_websocket.messages_sent) == 1
            confirmation = json.loads(mock_websocket.messages_sent[0])
            assert confirmation["type"] == "unsubscription_confirmed"
            assert confirmation["topic"] == "public:test"
            
            # Check connection is unsubscribed
            assert not connection.is_subscribed_to("public:test")
    
    @pytest.mark.asyncio
    async def test_handle_invalid_json_message(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test handling invalid JSON message."""
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_websocket, "valid-token")
            mock_websocket.messages_sent.clear()
            
            # Send invalid JSON
            await websocket_manager_instance.handle_message(connection.connection_id, "invalid json")
            
            # Check error response
            assert len(mock_websocket.messages_sent) == 1
            error_message = json.loads(mock_websocket.messages_sent[0])
            assert error_message["type"] == "error"
            assert error_message["error"]["code"] == "INVALID_JSON"
    
    @pytest.mark.asyncio
    async def test_broadcast_to_topic(self, websocket_manager_instance, test_user_session):
        """Test broadcasting message to topic subscribers."""
        # Create multiple mock WebSockets
        mock_ws1 = MockWebSocket()
        mock_ws2 = MockWebSocket()
        mock_ws3 = MockWebSocket()
        
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            # Connect multiple clients
            conn1 = await websocket_manager_instance.connect(mock_ws1, "token1")
            conn2 = await websocket_manager_instance.connect(mock_ws2, "token2")
            conn3 = await websocket_manager_instance.connect(mock_ws3, "token3")
            
            # Subscribe to topic
            topic = "public:broadcast-test"
            conn1.subscribe_to_topic(topic)
            conn2.subscribe_to_topic(topic)
            websocket_manager_instance._add_topic_subscription(topic, conn1.connection_id)
            websocket_manager_instance._add_topic_subscription(topic, conn2.connection_id)
            
            # Clear welcome messages
            mock_ws1.messages_sent.clear()
            mock_ws2.messages_sent.clear()
            mock_ws3.messages_sent.clear()
            
            # Broadcast message
            broadcast_message = {"type": "announcement", "message": "Hello everyone!"}
            await websocket_manager_instance.broadcast_to_topic(topic, broadcast_message)
            
            # Check only subscribed connections received the message
            assert len(mock_ws1.messages_sent) == 1
            assert len(mock_ws2.messages_sent) == 1
            assert len(mock_ws3.messages_sent) == 0  # Not subscribed
            
            # Check message content
            received1 = json.loads(mock_ws1.messages_sent[0])
            assert received1["type"] == "announcement"
            assert received1["message"] == "Hello everyone!"
            assert received1["topic"] == topic
            assert received1["broadcast"] is True
    
    @pytest.mark.asyncio
    async def test_send_to_user(self, websocket_manager_instance, test_user_session):
        """Test sending message to specific user."""
        mock_ws = MockWebSocket()
        
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_ws, "token")
            mock_ws.messages_sent.clear()
            
            # Send message to user
            user_message = {"type": "notification", "message": "Personal message"}
            await websocket_manager_instance.send_to_user(test_user_session.user_id, user_message)
            
            # Check message was received
            assert len(mock_ws.messages_sent) == 1
            received = json.loads(mock_ws.messages_sent[0])
            assert received["type"] == "notification"
            assert received["message"] == "Personal message"
            assert received["topic"] == f"user:{test_user_session.user_id}"
    
    @pytest.mark.asyncio
    async def test_send_to_group(self, websocket_manager_instance, test_user_session):
        """Test sending message to specific group."""
        mock_ws = MockWebSocket()
        
        with patch('app.shared.websocket_manager.auth_manager') as mock_auth:
            mock_auth.get_current_user.return_value = test_user_session
            
            connection = await websocket_manager_instance.connect(mock_ws, "token")
            mock_ws.messages_sent.clear()
            
            # Send message to group
            group_message = {"type": "group_notification", "message": "Group message"}
            await websocket_manager_instance.send_to_group("group1", group_message)
            
            # Check message was received (user is in group1)
            assert len(mock_ws.messages_sent) == 1
            received = json.loads(mock_ws.messages_sent[0])
            assert received["type"] == "group_notification"
            assert received["message"] == "Group message"
            assert received["topic"] == "group:group1"
    
    def test_validate_topic_access(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test topic access validation."""
        connection = WebSocketConnection(mock_websocket, test_user_session)
        
        # User can access their own topic
        assert websocket_manager_instance._validate_topic_access(
            connection, f"user:{test_user_session.user_id}"
        )
        
        # User cannot access other user's topic
        assert not websocket_manager_instance._validate_topic_access(
            connection, "user:other-user"
        )
        
        # User can access their group topics
        assert websocket_manager_instance._validate_topic_access(
            connection, "group:group1"
        )
        
        # User cannot access other group topics
        assert not websocket_manager_instance._validate_topic_access(
            connection, "group:other-group"
        )
        
        # User can access public topics
        assert websocket_manager_instance._validate_topic_access(
            connection, "public:announcements"
        )
        
        # User without admin permission cannot access admin topics
        assert not websocket_manager_instance._validate_topic_access(
            connection, "admin:system"
        )
        
        # User with admin permission can access admin topics
        admin_session = UserSession(
            session_id="admin-session",
            user_id="admin-user",
            groups=["admin"],
            permissions=["admin", "upload", "query"]
        )
        admin_connection = WebSocketConnection(mock_websocket, admin_session)
        assert websocket_manager_instance._validate_topic_access(
            admin_connection, "admin:system"
        )
    
    def test_get_connection_stats(self, websocket_manager_instance, mock_websocket, test_user_session):
        """Test getting connection statistics."""
        # Initially empty
        stats = websocket_manager_instance.get_connection_stats()
        assert stats["total_connections"] == 0
        assert stats["users_connected"] == 0
        assert stats["active_topics"] == 0
        
        # Add connection
        connection = WebSocketConnection(mock_websocket, test_user_session)
        websocket_manager_instance.connections[connection.connection_id] = connection
        websocket_manager_instance.user_connections[test_user_session.user_id] = {connection.connection_id}
        websocket_manager_instance.topic_subscriptions["test-topic"] = {connection.connection_id}
        
        # Check updated stats
        stats = websocket_manager_instance.get_connection_stats()
        assert stats["total_connections"] == 1
        assert stats["users_connected"] == 1
        assert stats["active_topics"] == 1
        assert stats["connections_by_user"][test_user_session.user_id] == 1
        assert stats["subscriptions_by_topic"]["test-topic"] == 1
    
    def test_health_check(self, websocket_manager_instance):
        """Test WebSocket manager health check."""
        health = websocket_manager_instance.health_check()
        
        assert "websocket_manager" in health
        assert "total_connections" in health
        assert "users_connected" in health
        assert "active_topics" in health
        assert "cleanup_task_running" in health
        assert "errors" in health
        assert isinstance(health["errors"], list)


@pytest.mark.asyncio
async def test_websocket_endpoint_integration():
    """Integration test for WebSocket endpoint."""
    from app.api.main import app
    from fastapi.testclient import TestClient
    
    # This would require more complex setup with actual WebSocket client
    # For now, we'll test that the endpoint exists and is properly configured
    client = TestClient(app)
    
    # Test that the WebSocket endpoint is registered
    # Note: TestClient doesn't support WebSocket testing well,
    # so this is a basic check that the endpoint exists
    assert any(route.path == "/ws/updates" for route in app.routes)