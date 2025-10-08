# Load Test Results

This directory contains the results from load testing runs.

## File Types

- `*.json` - Raw test data and metrics
- `*.html` - Generated HTML reports  
- `*.png` - Performance charts and graphs
- `*.log` - Test execution logs

## Result Files

Results are timestamped with the format: `YYYYMMDD_HHMMSS`

Example files:
- `load_test_results_20240315_143022.json`
- `stress_test_results_20240315_143022.json`
- `load_test_report_20240315_143022.html`

## Viewing Results

### HTML Reports
Open the HTML reports in your browser:
```bash
open results/load_test_report.html
```

### JSON Data
Raw JSON data can be analyzed programmatically:
```python
import json
with open('results/load_test_results_20240315_143022.json', 'r') as f:
    data = json.load(f)
    print(f"Average response time: {data['summary']['avg_response_time']}")
```

## Cleaning Up

To clean old results:
```bash
make clean
# or
rm -rf results/*.json results/*.html results/*.png
```

## Archiving

For long-term storage, consider archiving results:
```bash
tar -czf load_test_archive_$(date +%Y%m%d).tar.gz results/
```
