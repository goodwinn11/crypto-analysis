#!/bin/bash

# Deployment script for Crypto GPT-5
# Usage: ./deploy.sh

set -e

echo "🚀 Starting Crypto GPT-5 Deployment..."

# Configuration
REMOTE_USER="ubuntu"
REMOTE_HOST="40.172.225.59"
SSH_KEY="/Users/ivan/.ssh/Foodle.pem"
PROJECT_DIR="SravniAe"

# Function to run SSH commands
run_ssh() {
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "$1"
}

# Function to copy files
copy_file() {
    scp -i "$SSH_KEY" "$1" "$REMOTE_USER@$REMOTE_HOST:$2"
}

echo "📦 Copying files to server..."
# Copy important files
copy_file "crypto-gpt5.html" "~/$PROJECT_DIR/frontend/"
copy_file "backend_routes.py" "~/$PROJECT_DIR/"
copy_file "requirements.txt" "~/$PROJECT_DIR/"

echo "🔧 Updating backend routes..."
run_ssh "cd $PROJECT_DIR && cat backend_routes.py >> backend/app.py"

echo "🐳 Building Docker container..."
run_ssh "cd $PROJECT_DIR && sudo docker build -t sravniae-webapp -f Dockerfile.webapp ."

echo "♻️ Restarting services..."
run_ssh "sudo docker stop card-recommender-webapp || true"
run_ssh "sudo docker rm card-recommender-webapp || true"
run_ssh "cd $PROJECT_DIR && sudo docker run -d --name card-recommender-webapp -p 5000:5000 -v /home/ubuntu/$PROJECT_DIR/.env:/app/.env --restart unless-stopped sravniae-webapp"

echo "🔒 Checking SSL certificate..."
run_ssh "sudo certbot renew --dry-run || sudo certbot --nginx -d sravni.ae -d www.sravni.ae --non-interactive --agree-tos --email admin@sravni.ae"

echo "🔄 Reloading nginx..."
run_ssh "sudo nginx -t && sudo systemctl reload nginx"

echo "✅ Deployment complete!"
echo "📍 Access the application at: https://sravni.ae/crypto-gpt5"
echo "📊 API endpoint: https://sravni.ae/api/crypto/signals"