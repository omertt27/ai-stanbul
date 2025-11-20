#!/bin/bash
# 🔧 Infrastructure Setup Script for AI Istanbul
# This script sets up the production server from scratch

set -e

echo "🔧 AI Istanbul - Infrastructure Setup Script"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Update system
log_info "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
log_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    log_info "Docker installed"
else
    log_info "Docker already installed"
fi

# Install Docker Compose
log_info "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log_info "Docker Compose installed"
else
    log_info "Docker Compose already installed"
fi

# Install Nginx (for SSL termination)
log_info "Installing Nginx..."
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Install monitoring tools
log_info "Installing monitoring tools..."
sudo apt-get install -y htop iotop nethogs

# Create project directory
log_info "Creating project directory..."
sudo mkdir -p /opt/ai-istanbul
sudo mkdir -p /opt/ai-istanbul/backups
sudo mkdir -p /var/lib/ai-istanbul/postgres
sudo mkdir -p /var/lib/ai-istanbul/redis
sudo mkdir -p /var/log/ai-istanbul/backend
sudo chown -R $USER:$USER /opt/ai-istanbul

# Set up firewall
log_info "Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw --force enable

# Set up fail2ban
log_info "Installing fail2ban..."
sudo apt-get install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configure SSH (security hardening)
log_info "Hardening SSH configuration..."
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Set up SSL certificate
log_info "Setting up SSL certificate..."
read -p "Enter your domain name (e.g., ai-istanbul.com): " domain
sudo certbot certonly --nginx -d $domain -d www.$domain --non-interactive --agree-tos -m admin@$domain

# Set up automatic certificate renewal
log_info "Setting up automatic SSL renewal..."
(crontab -l 2>/dev/null; echo "0 0 * * * /usr/bin/certbot renew --quiet") | crontab -

# Install monitoring stack (Prometheus + Grafana)
log_info "Setting up monitoring stack..."
cat > /opt/ai-istanbul/docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "127.0.0.1:9090:9090"
    restart: always

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "127.0.0.1:3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=changeme
      - GF_SERVER_ROOT_URL=https://monitoring.ai-istanbul.com
    restart: always

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "127.0.0.1:9100:9100"
    restart: always

volumes:
  prometheus_data:
  grafana_data:
EOF

# Create Prometheus config
cat > /opt/ai-istanbul/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8002']
EOF

# Set up backup cron job
log_info "Setting up automated backups..."
cat > /opt/ai-istanbul/backup.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/opt/ai-istanbul/backups"

# Backup database
docker exec ai-istanbul-postgres-prod pg_dump -U $POSTGRES_USER ai_istanbul_prod | gzip > $BACKUP_DIR/db-$TIMESTAMP.sql.gz

# Backup Redis
docker exec ai-istanbul-redis-prod redis-cli --rdb /data/dump.rdb
docker cp ai-istanbul-redis-prod:/data/dump.rdb $BACKUP_DIR/redis-$TIMESTAMP.rdb

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete

# Upload to S3 (optional)
# aws s3 cp $BACKUP_DIR/ s3://ai-istanbul-backups/ --recursive
EOF

chmod +x /opt/ai-istanbul/backup.sh
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/ai-istanbul/backup.sh") | crontab -

# Create environment file template
log_info "Creating environment file template..."
cat > /opt/ai-istanbul/.env.template << 'EOF'
# Database
POSTGRES_USER=ai_istanbul_user
POSTGRES_PASSWORD=CHANGE_ME_STRONG_PASSWORD

# Redis
REDIS_PASSWORD=CHANGE_ME_STRONG_PASSWORD

# Application
SECRET_KEY=CHANGE_ME_SECRET_KEY
CORS_ORIGINS=https://ai-istanbul.com,https://www.ai-istanbul.com

# LLM Configuration
LLM_API_URL=http://localhost:8000/v1
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

# External APIs
OPENWEATHER_API_KEY=your_openweather_key
GOOGLE_MAPS_API_KEY=your_google_maps_key

# Monitoring
SENTRY_DSN=your_sentry_dsn
EOF

log_info "Infrastructure setup complete!"
echo ""
log_warn "⚠️  IMPORTANT: Please complete these manual steps:"
echo "1. Copy .env.template to .env and fill in all values"
echo "2. Review firewall rules: sudo ufw status"
echo "3. Set up DNS records for your domain"
echo "4. Configure monitoring at http://localhost:3001 (Grafana)"
echo "5. Test SSL certificate: sudo certbot certificates"
echo "6. Clone your repository to /opt/ai-istanbul"
echo ""
log_info "Next steps: Run ./scripts/deploy-production.sh to deploy the application"
