# AI Istanbul - Automated Backup Scheduling

## Setup Cron Jobs for Production

### Daily Database Backup (2 AM)
```bash
0 2 * * * /path/to/ai-stanbul/backup_script.sh
```

### Weekly Full System Backup (Sunday 3 AM)
```bash
0 3 * * 0 /path/to/ai-stanbul/full_backup_script.sh
```

### Backup Cleanup (Remove backups older than 30 days)
```bash
0 4 * * * find /path/to/ai-stanbul/backups -name "*.tar.gz" -mtime +30 -delete
```

## Monitoring Backup Success
```bash
# Add to monitoring system
/usr/local/bin/check_backup_status.sh
```
