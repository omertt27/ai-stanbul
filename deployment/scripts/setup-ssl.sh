#!/bin/bash
# SSL/TLS Certificate Setup Script for AI Istanbul (KAM)
# ⚠️  NOT NEEDED FOR VERCEL DEPLOYMENTS ⚠️
#
# This script is only for traditional VPS/server deployments.
# Vercel automatically provides SSL/TLS certificates for all domains.
#
# For Vercel deployment, see: VERCEL_DEPLOYMENT_GUIDE.md
#
# This script sets up Let's Encrypt SSL certificates using Certbot

set -e  # Exit on error

echo "⚠️  WARNING: This script is for VPS/server deployments only!"
echo "🔐 AI Istanbul SSL/TLS Certificate Setup"
echo "========================================"
echo ""
echo "If you're deploying to Vercel, you don't need this script."
echo "Vercel automatically handles SSL/TLS certificates."
echo ""
read -p "Continue anyway? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="${1:-yourdomain.com}"
WWW_DOMAIN="www.${DOMAIN}"
EMAIL="${2:-admin@${DOMAIN}}"
WEBROOT="/var/www/certbot"

echo -e "${YELLOW}Domain: ${DOMAIN}${NC}"
echo -e "${YELLOW}WWW Domain: ${WWW_DOMAIN}${NC}"
echo -e "${YELLOW}Email: ${EMAIL}${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (sudo)${NC}"
    exit 1
fi

# Step 1: Install Certbot
echo -e "${GREEN}Step 1: Installing Certbot...${NC}"
if ! command -v certbot &> /dev/null; then
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        apt-get update
        apt-get install -y certbot python3-certbot-nginx
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        yum install -y epel-release
        yum install -y certbot python3-certbot-nginx
    else
        echo -e "${RED}❌ Unsupported package manager. Please install certbot manually.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Certbot installed${NC}"
else
    echo -e "${GREEN}✅ Certbot already installed${NC}"
fi

# Step 2: Stop nginx temporarily
echo -e "${GREEN}Step 2: Stopping nginx...${NC}"
systemctl stop nginx || true

# Step 3: Obtain certificate
echo -e "${GREEN}Step 3: Obtaining SSL certificate...${NC}"
certbot certonly \
    --standalone \
    --non-interactive \
    --agree-tos \
    --email "${EMAIL}" \
    -d "${DOMAIN}" \
    -d "${WWW_DOMAIN}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ SSL certificate obtained successfully${NC}"
else
    echo -e "${RED}❌ Failed to obtain SSL certificate${NC}"
    exit 1
fi

# Step 4: Set up auto-renewal
echo -e "${GREEN}Step 4: Setting up auto-renewal...${NC}"

# Test renewal
certbot renew --dry-run

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Auto-renewal test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Auto-renewal test failed (will still work, but check configuration)${NC}"
fi

# Create renewal hook for nginx reload
cat > /etc/letsencrypt/renewal-hooks/deploy/nginx-reload.sh << 'EOF'
#!/bin/bash
systemctl reload nginx
EOF

chmod +x /etc/letsencrypt/renewal-hooks/deploy/nginx-reload.sh

echo -e "${GREEN}✅ Auto-renewal configured${NC}"

# Step 5: Update nginx configuration
echo -e "${GREEN}Step 5: Updating nginx configuration...${NC}"

# Backup existing configuration
if [ -f /etc/nginx/sites-available/ai-istanbul ]; then
    cp /etc/nginx/sites-available/ai-istanbul /etc/nginx/sites-available/ai-istanbul.backup
    echo -e "${GREEN}✅ Existing configuration backed up${NC}"
fi

# Update domain in configuration
sed -i "s/yourdomain.com/${DOMAIN}/g" /etc/nginx/sites-available/ai-istanbul

# Enable site
ln -sf /etc/nginx/sites-available/ai-istanbul /etc/nginx/sites-enabled/

# Test nginx configuration
nginx -t

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
else
    echo -e "${RED}❌ Nginx configuration is invalid${NC}"
    exit 1
fi

# Step 6: Start nginx
echo -e "${GREEN}Step 6: Starting nginx...${NC}"
systemctl start nginx
systemctl enable nginx

echo ""
echo -e "${GREEN}🎉 SSL/TLS Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Certificate Information:"
echo "  Location: /etc/letsencrypt/live/${DOMAIN}/"
echo "  Expiry: Check with 'certbot certificates'"
echo "  Auto-renewal: Configured (runs twice daily)"
echo ""
echo "Next Steps:"
echo "  1. Test your site: https://${DOMAIN}"
echo "  2. Check SSL rating: https://www.ssllabs.com/ssltest/analyze.html?d=${DOMAIN}"
echo "  3. Monitor certificate expiry: certbot certificates"
echo ""
echo "Useful Commands:"
echo "  - Renew certificate: certbot renew"
echo "  - Test renewal: certbot renew --dry-run"
echo "  - Check status: systemctl status certbot.timer"
echo ""
