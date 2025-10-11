# Keycloak Roles Setup Guide

## Issue Summary

Your Next.js frontend is now failing with authentication errors because:

1. **Token Expiration**: Tokens were expiring and not being refreshed before API calls
2. **Missing Roles**: Users need specific Keycloak roles to access certain endpoints

## Fixes Applied

### 1. Token Refresh (✅ FIXED)

Updated the following files to automatically refresh tokens before each API request:
- `/frontend/lib/api.ts` - Added token refresh logic to `performRequest()` and `uploadDocuments()`
- `/frontend/lib/authenticated-api.tsx` - Pass Keycloak instance to API client

**How it works:**
- Before each API request, the client checks if the token will expire within 30 seconds
- If expiring, it automatically refreshes using Keycloak's `updateToken()` method
- The refreshed token is used for the request

### 2. Required Roles Setup

The FastAPI backend requires specific roles for different operations:

#### Query Operations (`/api/query`)
**Required Roles** (user needs ONE of these):
- `query` - Basic query permission
- `standard` - Standard user with query access
- `admin` - Full administrative access

#### Upload Operations (`/api/documents/upload`)
**Required Roles** (user needs ONE of these):
- `upload` - Can upload documents
- `standard` - Standard user with upload access
- `admin` - Full administrative access

## How to Assign Roles in Keycloak

### Step 1: Access Keycloak Admin Console

1. Open browser and navigate to: **http://192.168.1.117:8080**
2. Click on "Administration Console"
3. Login with admin credentials

### Step 2: Navigate to Realm Roles

1. Select the **`rag_app`** realm (top-left dropdown)
2. Click **"Realm roles"** in the left sidebar

### Step 3: Create Required Roles (if not already created)

For each role that doesn't exist, click **"Create role"** and add:

**Role: `query`**
```
Role name: query
Description: Allows users to query documents
```

**Role: `standard`**
```
Role name: standard
Description: Standard user with query and upload permissions
```

**Role: `upload`**
```
Role name: upload
Description: Allows users to upload documents
```

**Role: `admin`**
```
Role name: admin
Description: Full administrative access
```

### Step 4: Assign Roles to User

1. Click **"Users"** in the left sidebar
2. Find and click on your user (search by username or email)
3. Click the **"Role mapping"** tab
4. Click **"Assign role"**
5. Select the roles you want to assign (recommended: `standard` or `admin`)
6. Click **"Assign"**

### Recommended Role Assignment

**For Regular Users:**
- Assign `standard` role (provides both query and upload access)

**For Administrators:**
- Assign `admin` role (provides full access)

**For Read-Only Users:**
- Assign `query` role only

**For Upload-Only Users:**
- Assign `upload` role only

## Testing After Role Assignment

### Step 1: Restart Frontend

Kill the current Next.js process and restart:
```bash
# Kill the process
kill 1489432

# Or kill all Next.js processes
pkill -f "next dev"

# Restart
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2/frontend
npm run dev
```

### Step 2: Clear Browser Cache & Logout

1. Open browser and navigate to your application
2. **Logout** from the application
3. **Clear browser cache** (Ctrl+Shift+Del)
4. **Close all browser tabs** for the application

### Step 3: Login Again

1. Navigate to the application
2. Login with your credentials
3. Keycloak will issue a new token with the assigned roles

### Step 4: Test Functionality

**Test Upload:**
1. Navigate to Upload page
2. Select a dataset/group
3. Choose a file
4. Click Upload
5. ✅ Should succeed without 401 error

**Test Query:**
1. Navigate to Query page
2. Enter a question
3. Submit query
4. ✅ Should succeed without 401 error

## Troubleshooting

### Still Getting 401 Errors?

**Check Token Claims:**
1. Open browser Developer Tools (F12)
2. Go to Console tab
3. Look for `[Auth]` log messages showing token parsed data
4. Verify that `realm_access.roles` includes your assigned roles

**Manually Decode Token:**
1. Open browser Developer Tools
2. Go to Application > Local Storage
3. Copy the token value
4. Go to https://jwt.io
5. Paste the token
6. Check the `realm_access.roles` array in the payload

### User Still Lacks Roles After Assignment?

1. **Force Logout in Keycloak:**
   - Admin Console > Users > [Your User] > Sessions tab
   - Click "Sign out" on all sessions

2. **Check Role Mapping:**
   - Admin Console > Users > [Your User] > Role mapping tab
   - Verify roles are listed under "Assigned roles"

3. **Check Client Roles:**
   - Some roles might be client-specific
   - Admin Console > Clients > fastapi-client > Roles tab
   - Make sure realm roles are available

### Token Refresh Not Working?

1. Check browser console for `[API]` log messages
2. Look for "Token refresh failed" warnings
3. If tokens can't refresh, user needs to logout and login again

## Backend Role Requirements Reference

From `/app/shared/models.py`:

```python
def has_permission(self, permission: str) -> bool:
    """Basic permission check mapped to Keycloak roles.
    
    - "query" requires role "query", "standard", or "admin".
    - Fallback: admin has all permissions.
    """
    role_set = set(self.roles or [])
    if 'admin' in role_set:
        return True
    if permission == 'query':
        return bool({'query', 'standard'} & role_set)
    return False
```

From `/app/api/main.py`:

```python
# Upload endpoint
@app.post("/api/documents/upload", tags=["Documents"])
async def upload_documents(
    ...
    current_user: User = Depends(require_any_role(["upload", "standard", "admin"])),
    ...
)

# Query endpoint
@app.post("/api/query", tags=["Query"])
async def submit_query(
    ...
    current_user: User = Depends(get_current_user),
    ...
):
    # Validate permissions
    if not current_user.has_permission("query"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have query permission"
        )
```

## Quick Fix Checklist

- [ ] Create required roles in Keycloak (`query`, `standard`, `upload`, `admin`)
- [ ] Assign `standard` or `admin` role to your user
- [ ] Logout from application
- [ ] Clear browser cache
- [ ] Login again
- [ ] Test upload functionality
- [ ] Test query functionality
- [ ] Check browser console for any remaining errors

## Summary

The application now has:
1. ✅ **Automatic token refresh** before every API call
2. ✅ **Proper multipart form data handling** for uploads
3. ⚠️  **Requires Keycloak roles** to be assigned to users

After assigning the appropriate roles in Keycloak and logging in again, both upload and query functionality should work without 401 errors.
