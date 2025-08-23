"""
WebSocket connection manager for real-time updates in the concurrent RAG system.
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import uuid

from .models import User
from .auth import auth_manager, TokenInvalidError, TokenExpiredError
from .redis_client import redis_client

# Set up logging
logger = logging.getLogger(__name__)


class WebSocketConnectionError(Exception):
    """Base exception for WebSocket connection errors."""
    pass


class WebSocketAuthenticationError(WebSocketConnectionError):
    """Raised when WebSocket authentication fails."""
    pass


class WebSocketConnection:
    """Represents a WebSocket connection with user context."""
    
    def __init__(self, websocket: WebSocket, user: User):
        """
        Initialize WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            user: Authenticated user object
        """
        self.connection_id = str(uuid.uuid4())
        self.websocket = websocket
        self.user = user
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.subscriptions: Set[str] = set()  # Topics this connection is subscribed to
        
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send message to WebSocket client.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            # Add metadata to message
            message_with_meta = {
                **message,
                "timestamp": datetime.now().isoformat(),
                "connection_id": self.connection_id
            }
            
            await self.websocket.send_text(json.dumps(message_with_meta))
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {self.connection_id}: {e}")
            return False
    
    async def send_error(self, error_code: str, error_message: str):
        """
        Send error message to WebSocket client.
        
        Args:
            error_code: Error code
            error_message: Error message
        """
        error_msg = {
            "type": "error",
            "error": {
                "code": error_code,
                "message": error_message
            }
        }
        await self.send_message(error_msg)
    
    def subscribe_to_topic(self, topic: str):
        """Subscribe connection to a topic."""
        self.subscriptions.add(topic)
        logger.debug(f"Connection {self.connection_id} subscribed to topic: {topic}")
    
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe connection from a topic."""
        self.subscriptions.discard(topic)
        logger.debug(f"Connection {self.connection_id} unsubscribed from topic: {topic}")
    
    def is_subscribed_to(self, topic: str) -> bool:
        """Check if connection is subscribed to a topic."""
        return topic in self.subscriptions
    
    def update_ping(self):
        """Update last ping timestamp."""
        self.last_ping = datetime.now()


class WebSocketManager:
    """
    Manages WebSocket connections with authentication and real-time messaging.
    """
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocketManager initialized")
    
    def authenticate_connection(self, token: str) -> User:
        """
        Authenticate WebSocket connection using JWT token.
        
        Args:
            token: JWT authentication token
            
        Returns:
            User: Authenticated user object
            
        Raises:
            WebSocketAuthenticationError: If authentication fails
        """
        try:
            payload = auth_manager.decode_token(token)
            # Normalize groups from either 'groups' claim or a single 'group' claim
            groups_claim = payload.get('groups') or payload.get('group') or []
            user = User(
                id=payload.get('sub'),
                username=payload.get('preferred_username'),
                email=payload.get('email'),
                groups=groups_claim,
                roles=payload.get('realm_access', {}).get('roles', [])
            )
            logger.info(f"WebSocket authentication successful for user {user.id}")
            return user
            
        except (TokenInvalidError, TokenExpiredError) as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            raise WebSocketAuthenticationError(f"Authentication failed: {e}")
        
        except Exception as e:
            logger.error(f"WebSocket authentication error: {e}")
            raise WebSocketAuthenticationError("Authentication service error")
    
    async def connect(self, websocket: WebSocket, token: str) -> WebSocketConnection:
        """
        Accept and authenticate WebSocket connection.
        """
        try:
            await websocket.accept()
            user = self.authenticate_connection(token)
            connection = WebSocketConnection(websocket, user)
            
            self.connections[connection.connection_id] = connection
            
            if user.id not in self.user_connections:
                self.user_connections[user.id] = set()
            self.user_connections[user.id].add(connection.connection_id)
            
            # Auto-subscribe to user-specific and group topics
            user_topic = f"user:{user.id}"
            connection.subscribe_to_topic(user_topic)
            self._add_topic_subscription(user_topic, connection.connection_id)
            
            for group in user.groups:
                group_topic = f"group:{group}"
                connection.subscribe_to_topic(group_topic)
                self._add_topic_subscription(group_topic, connection.connection_id)
            
            await connection.send_message({
                "type": "connection_established",
                "message": "WebSocket connection established successfully",
                "user_id": user.id,
                "connection_id": connection.connection_id,
                "subscriptions": list(connection.subscriptions)
            })
            
            logger.info(f"WebSocket connection established: {connection.connection_id} for user {user.id}")
            
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            return connection
            
        except WebSocketAuthenticationError:
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": {
                        "code": "AUTHENTICATION_FAILED",
                        "message": "WebSocket authentication failed"
                    }
                }))
                await websocket.close(code=4001)
            except Exception:
                pass
            raise
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
            raise WebSocketConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self, connection_id: str):
        """Disconnect and clean up WebSocket connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        try:
            user_id = connection.user.id
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            for topic in connection.subscriptions:
                self._remove_topic_subscription(topic, connection_id)
            
            del self.connections[connection_id]
            logger.info(f"WebSocket connection disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        connection = self.connections.get(connection_id)
        if not connection:
            logger.warning(f"Received message for unknown connection: {connection_id}")
            return
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                connection.update_ping()
                await connection.send_message({"type": "pong", "message": "pong"})
            
            elif message_type == "subscribe":
                topic = data.get("topic")
                if topic and self._validate_topic_access(connection, topic):
                    connection.subscribe_to_topic(topic)
                    self._add_topic_subscription(topic, connection_id)
                    await connection.send_message({"type": "subscription_confirmed", "topic": topic})
                else:
                    await connection.send_error("INVALID_TOPIC", "Invalid or unauthorized topic")
            
            elif message_type == "unsubscribe":
                topic = data.get("topic")
                if topic:
                    connection.unsubscribe_from_topic(topic)
                    self._remove_topic_subscription(topic, connection_id)
                    await connection.send_message({"type": "unsubscription_confirmed", "topic": topic})
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await connection.send_error("UNKNOWN_MESSAGE_TYPE", f"Unknown message type: {message_type}")
        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from connection {connection_id}")
            await connection.send_error("INVALID_JSON", "Invalid JSON message")
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await connection.send_error("MESSAGE_PROCESSING_ERROR", "Error processing message")
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any], exclude_connection: Optional[str] = None):
        """Broadcast message to all connections subscribed to a topic."""
        if topic not in self.topic_subscriptions:
            return
        
        connection_ids = self.topic_subscriptions[topic].copy()
        if exclude_connection:
            connection_ids.discard(exclude_connection)
        
        broadcast_message = {**message, "topic": topic, "broadcast": True}
        
        failed_connections = []
        for connection_id in connection_ids:
            connection = self.connections.get(connection_id)
            if connection:
                success = await connection.send_message(broadcast_message)
                if not success:
                    failed_connections.append(connection_id)
            else:
                failed_connections.append(connection_id)
        
        for failed_id in failed_connections:
            await self.disconnect(failed_id)
        
        logger.debug(f"Broadcasted message to topic {topic}: {len(connection_ids) - len(failed_connections)} successful, {len(failed_connections)} failed")
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections for a specific user."""
        user_topic = f"user:{user_id}"
        await self.broadcast_to_topic(user_topic, message)
    
    async def send_to_group(self, group_id: str, message: Dict[str, Any]):
        """Send message to all connections for a specific group."""
        group_topic = f"group:{group_id}"
        await self.broadcast_to_topic(group_topic, message)
    
    def _validate_topic_access(self, connection: WebSocketConnection, topic: str) -> bool:
        """Validate if connection has access to a topic."""
        if topic == f"user:{connection.user.id}":
            return True
        
        if topic.startswith("group:"):
            group_id = topic[6:]
            return group_id in connection.user.groups
        
        if topic.startswith("admin:"):
            return "admin" in connection.user.roles
        
        if topic.startswith("public:"):
            return True
        
        return False
    
    def _add_topic_subscription(self, topic: str, connection_id: str):
        """Add connection to topic subscription."""
        if topic not in self.topic_subscriptions:
            self.topic_subscriptions[topic] = set()
        self.topic_subscriptions[topic].add(connection_id)
    
    def _remove_topic_subscription(self, topic: str, connection_id: str):
        """Remove connection from topic subscription."""
        if topic in self.topic_subscriptions:
            self.topic_subscriptions[topic].discard(connection_id)
            if not self.topic_subscriptions[topic]:
                del self.topic_subscriptions[topic]
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale connections."""
        while True:
            try:
                await asyncio.sleep(300)
                stale_connections = [cid for cid, conn in self.connections.items() if (datetime.now() - conn.last_ping).total_seconds() > 600]
                for cid in stale_connections:
                    logger.info(f"Cleaning up stale WebSocket connection: {cid}")
                    await self.disconnect(cid)
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale WebSocket connections")
            except Exception as e:
                logger.error(f"Error during WebSocket cleanup: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.connections),
            "users_connected": len(self.user_connections),
            "active_topics": len(self.topic_subscriptions),
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()