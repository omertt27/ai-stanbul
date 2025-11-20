"""
Grafana Dashboard Configuration for AI Istanbul
Monitors LLM performance, cache hits, user feedback, and system health
"""

dashboards = {
    "llm_performance": {
        "title": "LLM Performance Dashboard",
        "panels": [
            {
                "title": "Response Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(llm_response_time_seconds_bucket[5m]))",
                        "legendFormat": "p95"
                    },
                    {
                        "expr": "histogram_quantile(0.50, rate(llm_response_time_seconds_bucket[5m]))",
                        "legendFormat": "p50"
                    }
                ]
            },
            {
                "title": "Requests Per Second",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(llm_requests_total[1m])",
                        "legendFormat": "{{language}}"
                    }
                ]
            },
            {
                "title": "Cache Hit Rate",
                "type": "stat",
                "targets": [
                    {
                        "expr": "rate(cache_hits_total[5m]) / rate(cache_requests_total[5m]) * 100"
                    }
                ]
            },
            {
                "title": "Token Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(llm_tokens_total[5m])",
                        "legendFormat": "Tokens/sec"
                    }
                ]
            },
            {
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(llm_errors_total[5m])",
                        "legendFormat": "{{error_type}}"
                    }
                ]
            },
            {
                "title": "Active Sessions",
                "type": "stat",
                "targets": [
                    {
                        "expr": "sum(active_sessions)"
                    }
                ]
            }
        ]
    },
    
    "cache_performance": {
        "title": "Cache Performance Dashboard",
        "panels": [
            {
                "title": "Cache Hit Rates by Tier",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(cache_l1_hits[5m]) / rate(cache_requests_total[5m]) * 100",
                        "legendFormat": "L1 (Memory)"
                    },
                    {
                        "expr": "rate(cache_l2_hits[5m]) / rate(cache_requests_total[5m]) * 100",
                        "legendFormat": "L2 (Semantic)"
                    },
                    {
                        "expr": "rate(cache_l3_hits[5m]) / rate(cache_requests_total[5m]) * 100",
                        "legendFormat": "L3 (Persistent)"
                    }
                ]
            },
            {
                "title": "Cache Misses",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(cache_misses_total[5m])"
                    }
                ]
            },
            {
                "title": "Redis Memory Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "redis_memory_used_bytes / redis_memory_max_bytes * 100"
                    }
                ]
            },
            {
                "title": "Cache Evictions",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(redis_evicted_keys_total[5m])"
                    }
                ]
            }
        ]
    },
    
    "user_feedback": {
        "title": "User Feedback & Satisfaction",
        "panels": [
            {
                "title": "Satisfaction Rate",
                "type": "stat",
                "targets": [
                    {
                        "expr": "sum(rate(feedback_positive_total[1h])) / (sum(rate(feedback_positive_total[1h])) + sum(rate(feedback_negative_total[1h]))) * 100"
                    }
                ]
            },
            {
                "title": "Feedback Over Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(feedback_positive_total[5m])",
                        "legendFormat": "Positive"
                    },
                    {
                        "expr": "rate(feedback_negative_total[5m])",
                        "legendFormat": "Negative"
                    }
                ]
            },
            {
                "title": "Average Rating",
                "type": "gauge",
                "targets": [
                    {
                        "expr": "avg(feedback_rating)"
                    }
                ]
            },
            {
                "title": "Feedback by Language",
                "type": "piechart",
                "targets": [
                    {
                        "expr": "sum by (language) (feedback_total)"
                    }
                ]
            }
        ]
    },
    
    "ab_testing": {
        "title": "A/B Testing Dashboard",
        "panels": [
            {
                "title": "Experiment Satisfaction Rates",
                "type": "graph",
                "targets": [
                    {
                        "expr": "sum by (variant) (rate(ab_test_positive_feedback[5m])) / (sum by (variant) (rate(ab_test_positive_feedback[5m])) + sum by (variant) (rate(ab_test_negative_feedback[5m]))) * 100"
                    }
                ]
            },
            {
                "title": "Requests by Variant",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(ab_test_requests[5m])",
                        "legendFormat": "{{variant}}"
                    }
                ]
            },
            {
                "title": "Response Time by Variant",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(ab_test_response_time_seconds_bucket[5m]))",
                        "legendFormat": "{{variant}}"
                    }
                ]
            }
        ]
    },
    
    "system_health": {
        "title": "System Health Dashboard",
        "panels": [
            {
                "title": "CPU Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)"
                    }
                ]
            },
            {
                "title": "Memory Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100"
                    }
                ]
            },
            {
                "title": "Disk I/O",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(node_disk_read_bytes_total[5m])",
                        "legendFormat": "Read"
                    },
                    {
                        "expr": "rate(node_disk_written_bytes_total[5m])",
                        "legendFormat": "Write"
                    }
                ]
            },
            {
                "title": "Network Traffic",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(node_network_receive_bytes_total[5m])",
                        "legendFormat": "Inbound"
                    },
                    {
                        "expr": "rate(node_network_transmit_bytes_total[5m])",
                        "legendFormat": "Outbound"
                    }
                ]
            },
            {
                "title": "HTTP Status Codes",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(http_requests_total[5m])",
                        "legendFormat": "{{status_code}}"
                    }
                ]
            }
        ]
    }
}


def generate_grafana_json():
    """Generate Grafana JSON dashboard configurations"""
    import json
    
    for dashboard_name, config in dashboards.items():
        filename = f"/Users/omer/Desktop/ai-stanbul/monitoring/grafana-{dashboard_name}.json"
        
        # Create Grafana-compatible JSON
        grafana_dashboard = {
            "dashboard": {
                "title": config["title"],
                "panels": config["panels"],
                "schemaVersion": 16,
                "version": 0,
                "refresh": "10s"
            },
            "overwrite": True
        }
        
        with open(filename, 'w') as f:
            json.dump(grafana_dashboard, f, indent=2)
        
        print(f"Generated: {filename}")


if __name__ == "__main__":
    generate_grafana_json()
