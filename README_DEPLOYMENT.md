# Crypto GPT-5 Deployment Guide

## Overview
This guide provides complete instructions for deploying the Crypto GPT-5 trading signals application with real-time analysis using the Williams Alligator-Fractal strategy.

## Architecture
- **Frontend**: HTML5 + JavaScript (crypto-gpt5.html)
- **Backend**: Python Flask API
- **Strategy**: Williams Alligator-Fractal with optional GPT analysis
- **Infrastructure**: Docker + Nginx + SSL

## Prerequisites
- Ubuntu server (tested on 24.04 LTS)
- Domain name pointed to server IP
- SSH access with sudo privileges

## Quick Deployment

### Option 1: Automated Deployment
```bash
./deploy.sh
```

### Option 2: Manual Deployment

#### 1. Server Setup
```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Install Nginx
sudo apt-get update && sudo apt-get install -y nginx

# Install Certbot for SSL
sudo apt-get install -y certbot python3-certbot-nginx
```

#### 2. Copy Files to Server
```bash
scp -i /path/to/key.pem crypto-gpt5.html ubuntu@server:/path/to/frontend/
scp -i /path/to/key.pem backend_routes.py ubuntu@server:/path/to/backend/
```

#### 3. Configure Backend
Add the routes from `backend_routes.py` to your Flask `app.py` file, ensuring they are placed BEFORE the `if __name__ == "__main__"` block.

#### 4. Build and Run Docker Container
```bash
# Build image
sudo docker build -t sravniae-webapp -f Dockerfile .

# Run container
sudo docker run -d \
  --name card-recommender-webapp \
  -p 5000:5000 \
  -v /path/to/.env:/app/.env \
  --restart unless-stopped \
  sravniae-webapp
```

#### 5. Configure Nginx
```bash
# Copy nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/sravniae
sudo ln -sf /etc/nginx/sites-available/sravniae /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

#### 6. Setup SSL Certificate
```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com \
  --non-interactive --agree-tos --email admin@yourdomain.com
```

## Environment Variables
Create a `.env` file with:
```env
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key

# Optional: Model selection
GPT_MODEL=gpt-4-turbo-preview

# Optional: Monitoring
CHECK_INTERVAL=300
OUTPUT_DIR=signal_reports
```

## API Endpoints

### Get Trading Signals
```bash
GET /api/crypto/signals?limit=30&use_gpt=true
```

**Parameters:**
- `limit`: Number of symbols to analyze (1-30)
- `use_gpt`: Enable GPT analysis (true/false)

**Response:**
```json
{
  "timestamp": "2025-08-28T14:35:51",
  "exchange": "binance",
  "timeframe": "4h",
  "symbols_analyzed": 30,
  "signals_found": 2,
  "signals": [
    {
      "symbol": "BTC/USDT",
      "direction": "LONG",
      "entry_price": 45000,
      "stop_loss": 44000,
      "take_profit": 47000,
      "risk_reward": 2.0,
      "gpt_analysis": {...}
    }
  ]
}
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
sudo docker logs card-recommender-webapp

# Common issues:
# - Port already in use: Change port in docker run command
# - Module import errors: Check requirements.txt
# - Syntax errors: Validate Python code
```

### SSL Certificate Issues
```bash
# Test certificate
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal
```

### API Returns 500 Error
```bash
# Check for JSON serialization issues
# Ensure all boolean values are converted to strings
# Check strategy.py has proper return statement in run_analysis()
```

## Monitoring

### Check Container Status
```bash
sudo docker ps
sudo docker logs --tail 50 card-recommender-webapp
```

### Check Nginx Access Logs
```bash
sudo tail -f /var/log/nginx/access.log
```

### Test Endpoints
```bash
# Test page
curl -I https://yourdomain.com/crypto-gpt5

# Test API
curl https://yourdomain.com/api/crypto/signals?limit=1
```

## Maintenance

### Update Code
```bash
# Pull latest changes
git pull origin main

# Rebuild container
sudo docker build -t sravniae-webapp .
sudo docker restart card-recommender-webapp
```

### Backup
```bash
# Backup .env and database
sudo cp .env .env.backup
sudo docker exec card-recommender-webapp tar czf /tmp/backup.tar.gz /app/data
sudo docker cp card-recommender-webapp:/tmp/backup.tar.gz ./
```

## Security Recommendations

1. **Firewall**: Configure UFW to only allow ports 22, 80, 443
2. **API Keys**: Never commit API keys to git
3. **Rate Limiting**: Implement rate limiting in nginx
4. **HTTPS Only**: Force HTTPS redirect
5. **Regular Updates**: Keep system and dependencies updated

## Support

For issues or questions:
- Check logs: `sudo docker logs card-recommender-webapp`
- Review nginx config: `sudo nginx -t`
- Verify SSL: `sudo certbot certificates`

## License
MIT License - See LICENSE file for details