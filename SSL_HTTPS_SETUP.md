# SSL/HTTPS Configuration for AI Istanbul

## SSL Certificate Setup

### 1. Let's Encrypt (Recommended for Production)
```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Generate SSL Certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### 2. FastAPI HTTPS Configuration
```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=443,
        ssl_keyfile="/path/to/private.key",
        ssl_certfile="/path/to/certificate.crt",
        ssl_ca_certs="/path/to/ca_bundle.crt"
    )
```

### 3. Nginx SSL Configuration
```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers (already implemented in FastAPI)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# HTTP to HTTPS Redirect
server {
    listen 80;
    listen [::]:80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

## SSL Certificate Renewal
```bash
# Add to crontab for automatic renewal
0 12 * * * /usr/bin/certbot renew --quiet
```

## Testing SSL Configuration
```bash
# Test SSL configuration
curl -I https://yourdomain.com

# Check SSL certificate
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
```
