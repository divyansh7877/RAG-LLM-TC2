#!/bin/bash
# Test Keycloak Configuration Script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

KEYCLOAK_URL="http://192.168.1.117:8080"
REALM="rag_app"
CLIENT_ID="fastapi-client"

echo "========================================="
echo "   Keycloak Configuration Test"
echo "========================================="
echo ""

# Test 1: Check if Keycloak is accessible
echo -e "${YELLOW}[TEST 1]${NC} Checking if Keycloak is accessible..."
if curl -f -s "${KEYCLOAK_URL}" > /dev/null; then
    echo -e "${GREEN}✓${NC} Keycloak is accessible at ${KEYCLOAK_URL}"
else
    echo -e "${RED}✗${NC} Cannot reach Keycloak at ${KEYCLOAK_URL}"
    echo "Make sure Keycloak is running and accessible"
    exit 1
fi

# Test 2: Check if realm exists
echo ""
echo -e "${YELLOW}[TEST 2]${NC} Checking if realm '${REALM}' exists..."
if curl -f -s "${KEYCLOAK_URL}/realms/${REALM}/.well-known/openid-configuration" > /dev/null; then
    echo -e "${GREEN}✓${NC} Realm '${REALM}' exists"
else
    echo -e "${RED}✗${NC} Realm '${REALM}' not found"
    echo "Create the realm in Keycloak admin console"
    exit 1
fi

# Test 3: Get OpenID configuration
echo ""
echo -e "${YELLOW}[TEST 3]${NC} Fetching OpenID configuration..."
CONFIG=$(curl -s "${KEYCLOAK_URL}/realms/${REALM}/.well-known/openid-configuration")

if [ -n "$CONFIG" ]; then
    echo -e "${GREEN}✓${NC} OpenID configuration retrieved successfully"
    
    # Extract and show important endpoints
    echo ""
    echo "Important endpoints:"
    echo "$CONFIG" | python3 -c "
import sys, json
config = json.load(sys.stdin)
print(f\"  Authorization: {config['authorization_endpoint']}\")
print(f\"  Token: {config['token_endpoint']}\")
print(f\"  UserInfo: {config['userinfo_endpoint']}\")
print(f\"  Logout: {config['end_session_endpoint']}\")
" 2>/dev/null || echo "  (Could not parse endpoints)"
else
    echo -e "${RED}✗${NC} Failed to retrieve OpenID configuration"
    exit 1
fi

# Test 4: Check if client can be authorized (will redirect to login)
echo ""
echo -e "${YELLOW}[TEST 4]${NC} Testing authorization endpoint..."
AUTH_URL="${KEYCLOAK_URL}/realms/${REALM}/protocol/openid-connect/auth?client_id=${CLIENT_ID}&redirect_uri=http://192.168.1.117:3000/&response_type=code&scope=openid"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$AUTH_URL")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
    echo -e "${GREEN}✓${NC} Authorization endpoint is working (HTTP $HTTP_CODE)"
    if [ "$HTTP_CODE" = "302" ]; then
        echo "  This is expected - it redirects to login page"
    fi
else
    echo -e "${RED}✗${NC} Authorization endpoint returned HTTP $HTTP_CODE"
    echo "Check client configuration in Keycloak"
fi

# Test 5: Check network accessibility from localhost
echo ""
echo -e "${YELLOW}[TEST 5]${NC} Checking if Keycloak is accessible from localhost..."
if curl -f -s --connect-timeout 5 "http://localhost:8080" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Keycloak is also accessible on localhost:8080"
    echo "  Consider using localhost for both frontend and Keycloak to avoid CORS issues"
elif curl -f -s --connect-timeout 5 "http://127.0.0.1:8080" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Keycloak is accessible on 127.0.0.1:8080"
    echo "  Consider using localhost for both frontend and Keycloak"
else
    echo -e "${GREEN}✓${NC} Keycloak is only on 192.168.1.117 (this is fine)"
    echo "  Make sure to access frontend via http://192.168.1.117:3000"
fi

echo ""
echo "========================================="
echo -e "${GREEN}✓ Keycloak Tests Complete${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Configure client in Keycloak admin console:"
echo "   URL: ${KEYCLOAK_URL}"
echo "   Realm: ${REALM}"
echo "   Client: ${CLIENT_ID}"
echo ""
echo "2. Required client settings:"
echo "   - Client authentication: OFF (public client)"
echo "   - Standard flow: ENABLED"
echo "   - Valid redirect URIs: http://192.168.1.117:3000/*"
echo "   - Web origins: http://192.168.1.117:3000"
echo ""
echo "3. Start the frontend:"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "4. Access via: http://192.168.1.117:3000"
echo ""