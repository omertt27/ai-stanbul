# 🔄 AI Istanbul Rollback Procedures Documentation

## 📋 **Overview**

This document provides comprehensive rollback procedures for the AI Istanbul system, ensuring rapid recovery from deployment issues, service failures, or critical bugs in production.

---

## 🎯 **Rollback Strategy Overview**

### 🏗️ **Multi-Layer Rollback Architecture**
1. **Application Layer**: Code rollback via Git and deployment tools
2. **Database Layer**: Schema and data rollback procedures
3. **Infrastructure Layer**: Service and configuration rollback
4. **Integration Layer**: Third-party service failover
5. **Frontend Layer**: Static asset and CDN rollback

### ⚡ **Rollback Speed Targets**
- **Critical Issues**: <5 minutes to safe state
- **Major Issues**: <15 minutes to previous version
- **Minor Issues**: <30 minutes to full rollback
- **Data Issues**: <60 minutes with data verification

---

## 🚨 **Emergency Rollback Procedures**

### 🔥 **Critical System Failure (Immediate Response)**

#### **1. Instant Traffic Redirect (0-2 minutes)**
```bash
#!/bin/bash
# Emergency traffic redirect script
# Run immediately if system is completely down

echo "🚨 EMERGENCY ROLLBACK: Redirecting traffic..."

# Option A: Load Balancer Rollback
curl -X POST "https://your-load-balancer/api/rollback" \
  -H "Authorization: Bearer $LB_API_TOKEN" \
  -d '{"target": "previous_version"}'

# Option B: DNS Rollback (if using DNS-based routing)
# Update DNS records to point to previous stable version
dig aistanbul.com  # Verify current DNS
# Manual DNS update required through provider

# Option C: CDN Rollback
curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/purge_cache" \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"purge_everything": true}'

echo "✅ Traffic redirected to safe endpoint"
```

#### **2. Service Shutdown (2-3 minutes)**
```bash
#!/bin/bash
# Immediate service shutdown script

echo "🛑 EMERGENCY: Stopping all services..."

# Stop application services
docker-compose down --remove-orphans

# Or if using systemd
sudo systemctl stop ai-istanbul-backend
sudo systemctl stop ai-istanbul-frontend

# Stop database writes (read-only mode)
sudo -u postgres psql -c "ALTER SYSTEM SET default_transaction_read_only = on;"
sudo systemctl reload postgresql

echo "✅ All services stopped - system in safe state"
```

#### **3. Maintenance Page Activation (3-5 minutes)**
```bash
#!/bin/bash
# Activate maintenance page

echo "🔧 Activating maintenance page..."

# Deploy static maintenance page
cp /opt/maintenance/maintenance.html /var/www/html/index.html

# Update load balancer to serve maintenance page
curl -X PUT "https://your-load-balancer/api/maintenance" \
  -H "Authorization: Bearer $LB_API_TOKEN" \
  -d '{"enabled": true, "message": "System under maintenance - back shortly"}'

echo "✅ Maintenance page active"
```

---

## 🔄 **Application Rollback Procedures**

### 🐙 **Git-Based Rollback**

#### **1. Identify Target Version**
```bash
# List recent releases
git log --oneline --decorate -10

# Find last stable version
git tag --sort=-version:refname | head -5

# Example output:
# v2.1.2  <- Current production (problematic)
# v2.1.1  <- Last known good version
# v2.1.0
# v2.0.9
# v2.0.8
```

#### **2. Execute Git Rollback**
```bash
#!/bin/bash
# Application rollback script

set -e

ROLLBACK_VERSION="v2.1.1"  # Last known good version
BACKUP_BRANCH="rollback-$(date +%Y%m%d-%H%M%S)"

echo "🔄 Starting application rollback to $ROLLBACK_VERSION..."

# 1. Create backup of current state
git checkout -b "$BACKUP_BRANCH"
git push origin "$BACKUP_BRANCH"

# 2. Rollback to target version
git checkout main
git reset --hard "$ROLLBACK_VERSION"

# 3. Force push (DANGEROUS - only in emergencies)
echo "⚠️  WARNING: Force pushing rollback..."
read -p "Continue? (y/N): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push --force-with-lease origin main
    echo "✅ Git rollback complete"
else
    echo "❌ Rollback cancelled"
    exit 1
fi
```

#### **3. Verify Rollback**
```bash
# Verify git state
git log --oneline -3
git describe --tags

# Check if rollback was successful
if git describe --tags | grep -q "$ROLLBACK_VERSION"; then
    echo "✅ Git rollback verified"
else
    echo "❌ Git rollback failed"
    exit 1
fi
```

---

### 🐳 **Docker-Based Rollback**

#### **1. Container Rollback**
```bash
#!/bin/bash
# Docker container rollback script

set -e

echo "🐳 Starting Docker rollback..."

# 1. Stop current containers
docker-compose down

# 2. Rollback to previous image version
PREVIOUS_TAG="v2.1.1"  # Last known good version

# Update docker-compose.yml or use specific tags
docker pull aistanbul/backend:$PREVIOUS_TAG
docker pull aistanbul/frontend:$PREVIOUS_TAG

# 3. Update docker-compose.yml
sed -i.bak "s|aistanbul/backend:latest|aistanbul/backend:$PREVIOUS_TAG|g" docker-compose.yml
sed -i.bak "s|aistanbul/frontend:latest|aistanbul/frontend:$PREVIOUS_TAG|g" docker-compose.yml

# 4. Start with previous version
docker-compose up -d

echo "✅ Docker rollback complete"
```

#### **2. Health Check After Rollback**
```bash
#!/bin/bash
# Post-rollback health check

echo "🔍 Verifying rollback health..."

# Wait for services to start
sleep 30

# Health checks
BACKEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/api/health)
FRONTEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)

if [ "$BACKEND_HEALTH" = "200" ] && [ "$FRONTEND_HEALTH" = "200" ]; then
    echo "✅ Rollback health check passed"
    exit 0
else
    echo "❌ Rollback health check failed"
    echo "Backend: $BACKEND_HEALTH, Frontend: $FRONTEND_HEALTH"
    exit 1
fi
```

---

## 🗄️ **Database Rollback Procedures**

### 📊 **Database Schema Rollback**

#### **1. Pre-Rollback Database Backup**
```bash
#!/bin/bash
# Emergency database backup before rollback

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="/backups/emergency_backup_$TIMESTAMP.sql"

echo "💾 Creating emergency database backup..."

# Create full database backup
pg_dump -h localhost -U istanbul_user -d istanbul_ai_prod > "$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"

# Verify backup
if [ -f "${BACKUP_FILE}.gz" ]; then
    echo "✅ Emergency backup created: ${BACKUP_FILE}.gz"
    echo "📊 Backup size: $(du -h ${BACKUP_FILE}.gz | cut -f1)"
else
    echo "❌ Backup creation failed"
    exit 1
fi
```

#### **2. Schema Migration Rollback**
```bash
#!/bin/bash
# Database migration rollback script

set -e

echo "🗄️ Starting database migration rollback..."

# Using Alembic (if using SQLAlchemy)
cd /opt/ai-istanbul/backend

# Get current migration version
CURRENT_REVISION=$(alembic current)
echo "Current revision: $CURRENT_REVISION"

# Get target revision (previous stable)
TARGET_REVISION="abc123def456"  # Replace with actual revision

# Rollback to target revision
alembic downgrade "$TARGET_REVISION"

# Verify rollback
NEW_REVISION=$(alembic current)
if [ "$NEW_REVISION" = "$TARGET_REVISION" ]; then
    echo "✅ Database migration rollback successful"
else
    echo "❌ Database migration rollback failed"
    exit 1
fi
```

#### **3. Data Restoration (if needed)**
```bash
#!/bin/bash
# Data restoration from backup

set -e

BACKUP_FILE="/backups/last_known_good_backup.sql.gz"

echo "📥 Restoring database from backup..."

# Stop application to prevent writes
systemctl stop ai-istanbul-backend

# Drop current database (DANGEROUS)
echo "⚠️  WARNING: About to drop current database"
read -p "Continue with data restoration? (y/N): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    
    # Create new database
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS istanbul_ai_prod;"
    sudo -u postgres psql -c "CREATE DATABASE istanbul_ai_prod;"
    
    # Restore from backup
    gunzip -c "$BACKUP_FILE" | sudo -u postgres psql istanbul_ai_prod
    
    echo "✅ Database restoration complete"
else
    echo "❌ Data restoration cancelled"
    exit 1
fi
```

---

## 🌐 **Frontend Rollback Procedures**

### 📦 **Static Asset Rollback**

#### **1. CDN Rollback**
```bash
#!/bin/bash
# Frontend CDN rollback script

set -e

echo "🌐 Starting frontend rollback..."

# 1. Rollback to previous build
PREVIOUS_BUILD="build-20250930-143022"  # Replace with actual build ID

# 2. Update CDN to serve previous build
if [ -d "/var/www/builds/$PREVIOUS_BUILD" ]; then
    
    # Update symlink to previous build
    ln -sfn "/var/www/builds/$PREVIOUS_BUILD" "/var/www/html/current"
    
    # Clear CDN cache
    curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/purge_cache" \
        -H "Authorization: Bearer $CF_API_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"purge_everything": true}'
    
    echo "✅ Frontend rollback complete"
else
    echo "❌ Previous build not found: $PREVIOUS_BUILD"
    exit 1
fi
```

#### **2. API Endpoint Rollback**
```bash
#!/bin/bash
# Update frontend to use previous API version

set -e

echo "🔗 Rolling back API endpoints..."

# Update frontend configuration to use previous API
FRONTEND_CONFIG="/var/www/html/current/config.js"
ROLLBACK_API_URL="https://api-v2-1-1.aistanbul.com"

# Backup current config
cp "$FRONTEND_CONFIG" "${FRONTEND_CONFIG}.backup"

# Update API URL in config
sed -i "s|API_BASE_URL.*|API_BASE_URL: '$ROLLBACK_API_URL',|g" "$FRONTEND_CONFIG"

echo "✅ Frontend API endpoints rolled back"
```

---

## 🔌 **Integration Rollback Procedures**

### 🤖 **AI Service Rollback**

#### **1. OpenAI API Fallback**
```bash
#!/bin/bash
# OpenAI API rollback to mock responses

echo "🤖 Activating AI service fallback..."

# Update environment to disable OpenAI and use mock responses
export OPENAI_API_KEY=""
export AI_FALLBACK_MODE="mock"
export MOCK_AI_RESPONSES="true"

# Restart backend with fallback configuration
systemctl restart ai-istanbul-backend

echo "✅ AI service rollback active - using mock responses"
```

#### **2. Google API Rollback**
```bash
#!/bin/bash
# Google API rollback to cached data

echo "🗺️ Activating Google API fallback..."

# Switch to cached/mock data mode
export GOOGLE_PLACES_API_KEY=""
export USE_MOCK_PLACES_DATA="true"
export CACHE_ONLY_MODE="true"

# Restart services
systemctl restart ai-istanbul-backend

echo "✅ Google API rollback active - using cached data"
```

---

## 📊 **Rollback Verification Procedures**

### ✅ **Post-Rollback Checklist**

#### **1. Service Health Verification**
```bash
#!/bin/bash
# Comprehensive rollback verification script

set -e

echo "🔍 Starting rollback verification..."

# 1. Service status checks
services=("ai-istanbul-backend" "ai-istanbul-frontend" "postgresql" "redis")
for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service"; then
        echo "✅ $service: Running"
    else
        echo "❌ $service: Not running"
        exit 1
    fi
done

# 2. HTTP endpoint checks
endpoints=(
    "http://localhost:8001/api/health"
    "http://localhost:3000"
    "http://localhost:8001/ai/chat"
)

for endpoint in "${endpoints[@]}"; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
    if [ "$status" = "200" ]; then
        echo "✅ $endpoint: HTTP $status"
    else
        echo "❌ $endpoint: HTTP $status"
        exit 1
    fi
done

# 3. Database connectivity
if psql -h localhost -U istanbul_user -d istanbul_ai_prod -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ Database: Connected"
else
    echo "❌ Database: Connection failed"
    exit 1
fi

# 4. Functional test
response=$(curl -s -X POST "http://localhost:8001/ai/chat" \
    -H "Content-Type: application/json" \
    -d '{"user_input": "test"}')

if echo "$response" | jq -e '.success' > /dev/null 2>&1; then
    echo "✅ AI Chat: Functional"
else
    echo "❌ AI Chat: Not functional"
    exit 1
fi

echo "🎉 Rollback verification completed successfully!"
```

#### **2. Performance Verification**
```bash
#!/bin/bash
# Performance verification after rollback

echo "⚡ Running performance verification..."

# Frontend load time test
start_time=$(date +%s%3N)
curl -s "http://localhost:3000" > /dev/null
end_time=$(date +%s%3N)
load_time=$((end_time - start_time))

if [ "$load_time" -lt 5000 ]; then
    echo "✅ Frontend load time: ${load_time}ms (acceptable)"
else
    echo "⚠️ Frontend load time: ${load_time}ms (slow)"
fi

# API response time test
start_time=$(date +%s%3N)
curl -s -X POST "http://localhost:8001/ai/chat" \
    -H "Content-Type: application/json" \
    -d '{"user_input": "quick test"}' > /dev/null
end_time=$(date +%s%3N)
api_time=$((end_time - start_time))

if [ "$api_time" -lt 15000 ]; then
    echo "✅ API response time: ${api_time}ms (acceptable)"
else
    echo "⚠️ API response time: ${api_time}ms (slow)"
fi
```

---

## 📋 **Rollback Decision Matrix**

### 🎯 **When to Rollback**

| Issue Severity | Response Time | Rollback Type | Approval Required |
|---------------|---------------|---------------|-------------------|
| **Critical** - System Down | Immediate | Emergency | No (Execute immediately) |
| **High** - Major Features Broken | <15 minutes | Application | Lead Developer |
| **Medium** - Performance Issues | <30 minutes | Partial | Product Owner |
| **Low** - Minor Bugs | <60 minutes | Hotfix | Team Decision |

### 🚨 **Escalation Procedures**

#### **Critical Issues (Immediate Rollback)**
- Complete system failure
- Security breaches
- Data corruption
- Payment system failures

#### **High Priority (Quick Rollback)**
- Major feature failures
- API outages affecting >50% of users
- Database connectivity issues
- Authentication system failures

#### **Medium Priority (Planned Rollback)**
- Performance degradation
- Non-critical feature issues
- UI/UX problems
- Minor integration failures

---

## 📞 **Emergency Contacts & Communication**

### 🆘 **Rollback Team**
```
Primary On-Call: [Name] - [Phone] - [Email]
Secondary On-Call: [Name] - [Phone] - [Email]
DevOps Lead: [Name] - [Phone] - [Email]
Database Admin: [Name] - [Phone] - [Email]
Product Owner: [Name] - [Phone] - [Email]
```

### 📢 **Communication Templates**

#### **Rollback Initiation Message**
```
🚨 ROLLBACK INITIATED
System: AI Istanbul
Severity: [Critical/High/Medium]
Rollback Type: [Emergency/Application/Database]
Estimated Downtime: [X minutes]
Initiated By: [Name]
Reason: [Brief description]
Status Updates: Every 10 minutes
```

#### **Rollback Completion Message**
```
✅ ROLLBACK COMPLETE
System: AI Istanbul
Rollback Duration: [X minutes]
Current Version: [Version]
Status: Fully operational
Next Steps: [Post-rollback actions]
Post-Mortem: [Scheduled time]
```

---

## 📖 **Rollback Playbooks**

### 📚 **Quick Reference Cards**

#### **Emergency Rollback (5-minute response)**
1. ⚡ Execute traffic redirect
2. 🛑 Stop problematic services
3. 🔧 Activate maintenance page
4. 📞 Notify stakeholders
5. 🔄 Begin detailed rollback

#### **Application Rollback (15-minute response)**
1. 📋 Identify rollback target
2. 💾 Create emergency backup
3. 🐙 Execute git rollback
4. 🐳 Update containers
5. ✅ Verify functionality

#### **Database Rollback (30-60 minute response)**
1. 🛑 Stop application writes
2. 💾 Create full backup
3. 🔄 Execute migration rollback
4. 📥 Restore data if needed
5. ✅ Verify data integrity

---

## 📊 **Rollback Metrics & Reporting**

### 📈 **Key Metrics to Track**
- **Rollback Frequency**: Number of rollbacks per month
- **Rollback Duration**: Time to complete rollback
- **Recovery Time**: Time to full functionality
- **Data Loss**: Amount of data lost (if any)
- **User Impact**: Number of affected users

### 📝 **Post-Rollback Report Template**
```
# Rollback Incident Report

## Summary
- **Date/Time**: [ISO timestamp]
- **Duration**: [Total rollback time]
- **Severity**: [Critical/High/Medium/Low]
- **Root Cause**: [Brief description]

## Timeline
- [HH:MM] Issue detected
- [HH:MM] Rollback initiated
- [HH:MM] Rollback completed
- [HH:MM] System verified operational

## Impact Assessment
- **Users Affected**: [Number]
- **Data Loss**: [None/Minimal/Significant]
- **Revenue Impact**: [Amount if applicable]
- **SLA Breach**: [Yes/No]

## Actions Taken
1. [Detailed list of rollback steps]
2. [Communication actions]
3. [Verification procedures]

## Lessons Learned
- **What Went Well**: [Positive aspects]
- **What Could Improve**: [Areas for improvement]
- **Action Items**: [Specific improvements to implement]

## Prevention Measures
- [Steps to prevent similar issues]
- [Process improvements]
- [Testing enhancements]
```

---

## ✅ **Rollback Readiness Checklist**

### 🛠️ **Pre-Production Preparation**
- [ ] **Backup Systems**: Automated backups configured and tested
- [ ] **Rollback Scripts**: All rollback scripts tested in staging
- [ ] **Documentation**: Rollback procedures documented and accessible
- [ ] **Team Training**: All team members trained on rollback procedures
- [ ] **Monitoring**: Alerts configured for rollback triggers
- [ ] **Communication**: Emergency communication channels established

### 🔄 **Regular Rollback Testing**
- [ ] **Monthly**: Test application rollback procedures
- [ ] **Quarterly**: Test database rollback procedures
- [ ] **Bi-annually**: Full disaster recovery simulation
- [ ] **Annually**: Review and update all rollback procedures

### 📋 **Production Rollback Readiness**
- [ ] **Version Control**: Clean git history with tagged releases
- [ ] **Build Artifacts**: Previous builds preserved and accessible
- [ ] **Database Backups**: Recent backups verified and tested
- [ ] **Infrastructure**: Rollback-capable deployment setup
- [ ] **Monitoring**: Real-time system health monitoring active

---

**Rollback Procedures Status: COMPLETE ✅**  
**Last Updated:** October 2, 2025  
**Next Review:** Monthly rollback procedure review and testing  
**Emergency Hotline:** [Emergency contact information]
