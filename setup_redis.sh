#!/bin/bash

# Redis installation and setup script for Ubuntu/Debian

echo "Setting up Redis for Concurrent RAG System..."

# Check if Redis is already installed
if command -v redis-server &> /dev/null; then
    echo "Redis is already installed."
    redis-server --version
else
    echo "Installing Redis..."
    
    # Update package list
    sudo apt update
    
    # Install Redis
    sudo apt install -y redis-server
    
    # Check installation
    if command -v redis-server &> /dev/null; then
        echo "Redis installed successfully!"
        redis-server --version
    else
        echo "Redis installation failed. Please install manually."
        exit 1
    fi
fi

# Configure Redis for development
echo "Configuring Redis..."

# Create Redis configuration directory if it doesn't exist
sudo mkdir -p /etc/redis

# Basic Redis configuration for development
sudo tee /etc/redis/redis.conf > /dev/null <<EOF
# Redis configuration for Concurrent RAG System
port 6379
bind 127.0.0.1
timeout 0
tcp-keepalive 300
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
maxmemory 256mb
maxmemory-policy allkeys-lru
EOF

# Start and enable Redis service
echo "Starting Redis service..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Check Redis status
if sudo systemctl is-active --quiet redis-server; then
    echo "Redis is running successfully!"
    
    # Test Redis connection
    if redis-cli ping | grep -q "PONG"; then
        echo "Redis connection test successful!"
    else
        echo "Redis connection test failed."
    fi
else
    echo "Failed to start Redis service."
    exit 1
fi

echo "Redis setup completed!"
echo ""
echo "You can now:"
echo "1. Start the application with: python -m app.api.main"
echo "2. Start Celery workers with: celery -A app.workers.celery_app worker --loglevel=info"
echo "3. Monitor Redis with: redis-cli monitor"