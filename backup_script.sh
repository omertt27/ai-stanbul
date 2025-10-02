#!/bin/bash
# AI Istanbul - Database Backup Script
# This script should be customized for your production database

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups"
DB_NAME="ai_istanbul"

# PostgreSQL backup (customize for your setup)
# pg_dump $DB_NAME > $BACKUP_DIR/db_backup_$DATE.sql

# Redis backup (if using Redis persistence)
# redis-cli BGSAVE

# Application state backup
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz \
    --exclude=node_modules \
    --exclude=__pycache__ \
    --exclude=.git \
    --exclude=backups \
    .

echo "Backup completed: $BACKUP_DIR/app_backup_$DATE.tar.gz"
