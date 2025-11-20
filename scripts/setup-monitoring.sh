#!/bin/bash

# Setup Monitoring Stack for AI Istanbul
# Prometheus + Grafana + Loki + AlertManager

set -e

echo "🔧 Setting up monitoring stack for AI Istanbul..."

# Create monitoring directory
mkdir -p monitoring/grafana-dashboards

# Create Prometheus configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'ai-istanbul'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Load rules
rule_files:
  - "alerts.yml"

# Scrape configurations
scrape_configs:
  # Backend API
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8002']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Frontend
  - job_name: 'frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Node metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Create alert rules
cat > monitoring/alerts.yml << 'EOF'
groups:
  - name: ai_istanbul_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # Slow response time
      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(llm_response_time_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow LLM response time"
          description: "95th percentile response time is {{ $value }} seconds"

      # High memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      # Cache hit rate too low
      - alert: LowCacheHitRate
        expr: rate(cache_hits_total[5m]) / rate(cache_requests_total[5m]) < 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      # Service down
      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.job }} has been down for more than 2 minutes"

      # Low satisfaction rate
      - alert: LowSatisfactionRate
        expr: sum(rate(feedback_positive_total[1h])) / (sum(rate(feedback_positive_total[1h])) + sum(rate(feedback_negative_total[1h]))) < 0.6
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Low user satisfaction rate"
          description: "Satisfaction rate is {{ $value | humanizePercentage }}"
EOF

# Create Grafana datasources
cat > monitoring/grafana-datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true

  - name: Redis
    type: redis-datasource
    access: proxy
    url: redis://redis:6379
    editable: true
EOF

# Create Loki configuration
cat > monitoring/loki-config.yml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-05-15
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /loki/index
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF

# Create Promtail configuration
cat > monitoring/promtail-config.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: backend
    static_configs:
      - targets:
          - localhost
        labels:
          job: backend
          __path__: /var/log/backend/*.log

  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*.log
EOF

# Create AlertManager configuration
cat > monitoring/alertmanager.yml << 'EOF'
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

receivers:
  - name: 'default'
    # Add your notification channels here
    # Example: Slack, PagerDuty, Email, etc.
    webhook_configs:
      - url: 'http://localhost:5001/alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
EOF

# Generate Grafana dashboards
echo "📊 Generating Grafana dashboards..."
python3 monitoring/grafana_dashboards.py

# Set permissions
chmod +x monitoring/*.yml

echo "✅ Monitoring configuration complete!"
echo ""
echo "📋 Next steps:"
echo "1. Start monitoring stack:"
echo "   docker-compose -f docker-compose.monitoring.yml up -d"
echo ""
echo "2. Access dashboards:"
echo "   Grafana:     http://localhost:3001 (admin/admin123)"
echo "   Prometheus:  http://localhost:9090"
echo "   AlertManager: http://localhost:9093"
echo ""
echo "3. Configure alerts in monitoring/alerts.yml"
echo "4. Add notification channels in monitoring/alertmanager.yml"
echo ""
echo "🔐 Security: Change default Grafana password!"
