# Design Document

## Overview

The authentication system fix addresses critical issues in the current multi-user RAG system where users cannot successfully log in. This design creates a unified, secure authentication system that provides consistent login functionality for the web frontend.

The solution centers on establishing FastAPI as the primary authentication provider with a web-based frontend interface, ensuring a seamless and reliable login experience.

## Architecture

### Current State Analysis
- **Problem**: Authentication system not functioning properly for web frontend users
- **Impact**: Users cannot access the RAG system despite having valid credentials
- **Root Cause**: Incomplete or broken authentication implementation in the current system

### Proposed Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │───▶│  FastAPI Auth   │───▶│  Redis Session  │
│   Login Page    │    │   Controller    │    │    Storage      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         └─────────────▶│  Protected      │◀─────────────┘
                        │  Web Interface  │
                        └─────────────────┘
```

### Design Decisions

1. **FastAPI as Authentication Authority**: FastAPI will handle all authentication logic, providing a centralized and secure authentication mechanism
   - **Rationale**: FastAPI offers better security middleware, session management, and API standardization

2. **Redis-Based Session Management**: All user sessions will be stored in Redis for consistency and scalability
   - **Rationale**: Enables centralized session state management, supports horizontal scaling

3. **Web Frontend Integration**: The web frontend will validate sessions through FastAPI endpoints
   - **Rationale**: Creates a clean separation between authentication logic and UI presentation

## Components and Interfaces

### Authentication Controller (`app/shared/auth.py`)
**Purpose**: Centralized authentication logic and session management

**Key Methods**:
- `authenticate_user(username: str, password: str) -> UserSession`
- `create_session(user: User) -> str`
- `validate_session(session_token: str) -> Optional[UserSession]`
- `logout_user(session_token: str) -> bool`

**Interfaces**:
- Input: User credentials (username/password)
- Output: Session tokens, user objects, authentication status
- Dependencies: Redis client, user database, password hashing utilities

### Session Manager (`app/shared/session_manager.py`)
**Purpose**: Handle session lifecycle and validation

**Key Methods**:
- `store_session(session_token: str, user_data: dict, ttl: int)`
- `get_session(session_token: str) -> Optional[dict]`
- `invalidate_session(session_token: str)`
- `cleanup_expired_sessions()`

**Interfaces**:
- Storage: Redis with configurable TTL
- Security: Secure token generation, session data encryption

### FastAPI Authentication Endpoints (`app/api/main.py`)
**New Endpoints**:
- `POST /auth/login` - User authentication
- `POST /auth/logout` - Session termination
- `GET /auth/validate` - Session validation
- `GET /auth/user` - Current user information

**Middleware Integration**:
- Authentication middleware for protected routes
- CORS configuration for web frontend integration
- Rate limiting for login attempts

### Web Frontend Integration Layer
**Purpose**: Bridge web frontend with FastAPI authentication

**Implementation**:
- Session validation before web interface access
- Redirect mechanism for unauthenticated users
- User context passing to frontend components

## Data Models

### User Model
```python
class User(BaseModel):
    user_id: str
    username: str
    password_hash: str
    group_id: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
```

### UserSession Model
```python
class UserSession(BaseModel):
    session_token: str
    user_id: str
    username: str
    group_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool
```

### AuthRequest Model
```python
class AuthRequest(BaseModel):
    username: str
    password: str
```

### AuthResponse Model
```python
class AuthResponse(BaseModel):
    success: bool
    session_token: Optional[str]
    user: Optional[User]
    message: str
```

## Error Handling

### Authentication Errors
- **Invalid Credentials**: Clear error message without revealing whether username or password is incorrect
- **Account Locked**: Rate limiting response with retry information
- **Session Expired**: Automatic redirect to login with session expiry message
- **System Errors**: Generic error message with detailed logging for debugging

### Error Response Format
```python
class AuthError(BaseModel):
    error_code: str
    message: str
    details: Optional[dict]
    retry_after: Optional[int]  # For rate limiting
```

### Logging Strategy
- **Authentication Attempts**: Log all login attempts with IP, timestamp, and outcome
- **Session Events**: Track session creation, validation, and expiration
- **Security Events**: Failed login attempts, suspicious activity patterns
- **System Errors**: Detailed error context for debugging

## Testing Strategy

### Unit Tests
- **Authentication Logic**: Test credential validation, password hashing, session creation
- **Session Management**: Test session storage, retrieval, expiration, cleanup
- **Error Handling**: Test all error scenarios and response formats
- **Security Functions**: Test rate limiting, token generation, encryption

### Integration Tests
- **FastAPI Endpoints**: Test all authentication endpoints with various scenarios
- **Web Frontend Integration**: Test session validation and user context passing
- **Redis Integration**: Test session storage and retrieval operations
- **Cross-Application Flow**: Test complete login flow from web frontend through FastAPI

### Security Tests
- **Password Security**: Test password hashing and validation
- **Session Security**: Test token generation, validation, and expiration
- **Rate Limiting**: Test login attempt throttling
- **Authorization**: Test user access control and group permissions

### User Experience Tests
- **Login Flow**: Test complete user login experience
- **Error Feedback**: Test error message clarity and helpfulness
- **Session Persistence**: Test session behavior across browser sessions
- **Logout Process**: Test session cleanup and redirect behavior

## Security Considerations

### Password Security
- **Hashing**: Use bcrypt or Argon2 for password hashing
- **Salt**: Individual salts for each password
- **Complexity**: Enforce minimum password requirements

### Session Security
- **Token Generation**: Cryptographically secure random tokens
- **Storage**: Encrypted session data in Redis
- **Expiration**: Configurable session timeouts
- **Invalidation**: Secure logout and session cleanup

### Rate Limiting
- **Login Attempts**: Limit failed login attempts per IP/username
- **Progressive Delays**: Increasing delays for repeated failures
- **Account Lockout**: Temporary account lockout after threshold

### Data Protection
- **Input Validation**: Sanitize all authentication inputs
- **SQL Injection**: Use parameterized queries
- **XSS Protection**: Escape user data in responses
- **CSRF Protection**: Implement CSRF tokens for state-changing operations

## Performance Considerations

### Session Management
- **Redis Optimization**: Use appropriate data structures and expiration policies
- **Connection Pooling**: Efficient Redis connection management
- **Caching**: Cache frequently accessed user data

### Authentication Flow
- **Minimal Database Queries**: Optimize user lookup and validation
- **Async Operations**: Use FastAPI's async capabilities for I/O operations
- **Response Times**: Target sub-200ms authentication response times

### Scalability
- **Stateless Design**: Authentication logic independent of server instance
- **Horizontal Scaling**: Session storage supports multiple application instances
- **Load Distribution**: Authentication load can be distributed across workers