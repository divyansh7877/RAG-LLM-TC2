#!/usr/bin/env python3
"""
Simple verification test for authentication endpoints.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_auth_endpoints():
    """Test authentication endpoints manually."""
    print("Testing authentication endpoints...")
    
    # Test health check first
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health status: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test registration
    try:
        registration_data = {
            "username": "testuser123",
            "password": "testpassword123",
            "groups": ["assistance"]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/auth/register",
            json=registration_data
        )
        print(f"Registration: {response.status_code}")
        if response.status_code == 200:
            print(f"Registration response: {response.json()}")
        else:
            print(f"Registration error: {response.text}")
    except Exception as e:
        print(f"Registration failed: {e}")
    
    # Test login
    try:
        login_data = {
            "username": "assistant1",  # Use existing user
            "password": "password1"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json=login_data
        )
        print(f"Login: {response.status_code}")
        if response.status_code == 200:
            login_result = response.json()
            print(f"Login successful: {login_result['user_id']}")
            token = login_result['access_token']
            
            # Test session info
            headers = {"Authorization": f"Bearer {token}"}
            session_response = requests.get(
                f"{BASE_URL}/api/auth/session",
                headers=headers
            )
            print(f"Session info: {session_response.status_code}")
            if session_response.status_code == 200:
                print(f"Session: {session_response.json()}")
            
            # Test token validation
            validate_response = requests.post(
                f"{BASE_URL}/api/auth/validate",
                headers=headers
            )
            print(f"Token validation: {validate_response.status_code}")
            
            # Test logout
            logout_response = requests.post(
                f"{BASE_URL}/api/auth/logout",
                headers=headers
            )
            print(f"Logout: {logout_response.status_code}")
            
        else:
            print(f"Login error: {response.text}")
    except Exception as e:
        print(f"Login failed: {e}")
    
    return True

if __name__ == "__main__":
    print("Authentication endpoints verification")
    print("=" * 50)
    test_auth_endpoints()