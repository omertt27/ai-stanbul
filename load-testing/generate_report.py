"""
AI Istanbul Load Testing Report Generator
Generate comprehensive HTML reports from test results
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console

console = Console()

class ReportGenerator:
    """Generate comprehensive test reports"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def find_latest_results(self) -> Dict[str, str]:
        """Find the latest test result files"""
        
        result_files = {
            'load_test': None,
            'stress_test': None,
            'endurance_test': None,
            'integration_test': None,
            'frontend_test': None
        }
        
        # Search for result files
        for test_type in result_files.keys():
            pattern = f"{test_type}_results_*.json"
            files = glob.glob(pattern)
            if files:
                # Get the latest file
                latest_file = max(files, key=os.path.getctime)
                result_files[test_type] = latest_file
        
        return result_files
    
    def load_test_results(self, result_files: Dict[str, str]) -> Dict[str, Dict]:
        """Load test results from JSON files"""
        
        results = {}
        
        for test_type, file_path in result_files.items():
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        results[test_type] = json.load(f)
                    console.print(f"[green]✅ Loaded {test_type} results from {file_path}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to load {file_path}: {e}[/red]")
            else:
                console.print(f"[yellow]⚠️ No results found for {test_type}[/yellow]")
        
        return results
    
    def generate_performance_charts(self, results: Dict[str, Dict]) -> List[str]:
        """Generate performance charts"""
        
        chart_files = []
        
        # Response time comparison chart
        if any(test_type in results for test_type in ['load_test', 'stress_test', 'endurance_test']):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            test_names = []
            avg_times = []
            p95_times = []
            
            for test_type in ['load_test', 'stress_test', 'endurance_test']:
                if test_type in results:
                    result = results[test_type]
                    
                    if 'average_response_time' in result:
                        test_names.append(test_type.replace('_', ' ').title())
                        avg_times.append(result['average_response_time'])
                        p95_times.append(result.get('p95_response_time', 0))
                    elif 'results' in result and 'avg_response_time_ms' in result['results']:
                        test_names.append(test_type.replace('_', ' ').title())
                        avg_times.append(result['results']['avg_response_time_ms'])
                        p95_times.append(result['results'].get('p95_response_time', 0))
            
            if test_names:
                x = range(len(test_names))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], avg_times, width, label='Average Response Time', alpha=0.8)
                ax.bar([i + width/2 for i in x], p95_times, width, label='95th Percentile', alpha=0.8)
                
                ax.set_xlabel('Test Type')
                ax.set_ylabel('Response Time (ms)')
                ax.set_title('Response Time Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(test_names)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                chart_file = os.path.join(self.output_dir, 'response_time_comparison.png')
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files.append(chart_file)
        
        # Throughput comparison
        if any(test_type in results for test_type in ['load_test', 'stress_test']):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            test_names = []
            throughputs = []
            
            for test_type in ['load_test', 'stress_test']:
                if test_type in results:
                    result = results[test_type]
                    
                    if 'requests_per_second' in result:
                        test_names.append(test_type.replace('_', ' ').title())
                        throughputs.append(result['requests_per_second'])
                    elif 'results' in result and 'requests_per_second' in result['results']:
                        test_names.append(test_type.replace('_', ' ').title())
                        throughputs.append(result['results']['requests_per_second'])
            
            if test_names:
                bars = ax.bar(test_names, throughputs, alpha=0.8, color=['#2E86AB', '#A23B72'])
                ax.set_ylabel('Requests per Second')
                ax.set_title('Throughput Comparison')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, throughputs):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}', ha='center', va='bottom')
                
                chart_file = os.path.join(self.output_dir, 'throughput_comparison.png')
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files.append(chart_file)
        
        # Error rate comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        test_names = []
        error_rates = []
        
        for test_type in ['load_test', 'stress_test', 'endurance_test', 'integration_test']:
            if test_type in results:
                result = results[test_type]
                
                error_rate = 0
                if 'error_rate' in result:
                    error_rate = result['error_rate']
                elif 'results' in result and 'error_rate' in result['results']:
                    error_rate = result['results']['error_rate']
                elif test_type == 'integration_test':
                    # Calculate error rate from integration test
                    total_steps = sum(r.get('total_steps', 0) for r in result.values() if isinstance(r, dict))
                    failed_steps = sum(r.get('failed_steps', 0) for r in result.values() if isinstance(r, dict))
                    error_rate = (failed_steps / total_steps * 100) if total_steps > 0 else 0
                
                test_names.append(test_type.replace('_', ' ').title())
                error_rates.append(error_rate)
        
        if test_names:
            colors = ['green' if rate < 1 else 'orange' if rate < 5 else 'red' for rate in error_rates]
            bars = ax.bar(test_names, error_rates, alpha=0.8, color=colors)
            ax.set_ylabel('Error Rate (%)')
            ax.set_title('Error Rate Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, error_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.1f}%', ha='center', va='bottom')
            
            chart_file = os.path.join(self.output_dir, 'error_rate_comparison.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart_file)
        
        # Frontend performance chart
        if 'frontend_test' in results:
            frontend_result = results['frontend_test']
            pages_tested = frontend_result.get('pages_tested', [])
            
            if pages_tested:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Page load times
                page_names = [p['name'] for p in pages_tested]
                load_times = [p['desktop_metrics']['page_load_time'] for p in pages_tested]
                mobile_times = [p['mobile_metrics']['page_load_time'] for p in pages_tested]
                
                x = range(len(page_names))
                width = 0.35
                
                ax1.bar([i - width/2 for i in x], load_times, width, label='Desktop', alpha=0.8)
                ax1.bar([i + width/2 for i in x], mobile_times, width, label='Mobile', alpha=0.8)
                ax1.set_xlabel('Page')
                ax1.set_ylabel('Load Time (ms)')
                ax1.set_title('Page Load Times')
                ax1.set_xticks(x)
                ax1.set_xticklabels(page_names, rotation=45)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Bundle sizes
                all_bundles = {}
                for page in pages_tested:
                    for bundle, size in page['desktop_metrics']['bundle_sizes'].items():
                        if bundle not in all_bundles:
                            all_bundles[bundle] = size
                        else:
                            all_bundles[bundle] = max(all_bundles[bundle], size)
                
                if all_bundles:
                    bundle_names = list(all_bundles.keys())[:10]  # Top 10 bundles
                    bundle_sizes = [all_bundles[name]/1024 for name in bundle_names]  # Convert to KB
                    
                    ax2.barh(bundle_names, bundle_sizes, alpha=0.8)
                    ax2.set_xlabel('Size (KB)')
                    ax2.set_title('Bundle Sizes')
                    ax2.grid(True, alpha=0.3)
                
                chart_file = os.path.join(self.output_dir, 'frontend_performance.png')
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files.append(chart_file)
        
        return chart_files
    
    def generate_html_report(self, results: Dict[str, Dict], chart_files: List[str]) -> str:
        """Generate comprehensive HTML report"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Istanbul Load Testing Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            border-radius: 5px;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 1.1em;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-unit {{
            font-size: 0.8em;
            color: #7f8c8d;
        }}
        .status-good {{
            color: #27ae60 !important;
            border-left-color: #27ae60 !important;
        }}
        .status-warning {{
            color: #f39c12 !important;
            border-left-color: #f39c12 !important;
        }}
        .status-error {{
            color: #e74c3c !important;
            border-left-color: #e74c3c !important;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .test-summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .test-summary h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .recommendations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .recommendations ul {{
            color: #856404;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 AI Istanbul Load Testing Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
"""
        
        # Executive Summary
        html_content += """
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="test-summary">
"""
        
        # Calculate overall metrics
        total_tests = len([k for k in results.keys() if results[k]])
        successful_tests = 0
        
        overall_metrics = {
            'avg_response_time': 0,
            'max_throughput': 0,
            'min_error_rate': 100,
            'total_requests': 0
        }
        
        response_times = []
        error_rates = []
        throughputs = []
        
        for test_type, result in results.items():
            if not result:
                continue
                
            if test_type in ['load_test', 'stress_test', 'endurance_test']:
                # Extract metrics based on result structure
                if 'results' in result:
                    res = result['results']
                    # Handle case where results is a list of scenarios
                    if isinstance(res, list) and len(res) > 0:
                        res = res[0]  # Use first scenario
                    
                    if isinstance(res, dict) and res.get('error_rate', 100) < 10:  # Less than 10% error rate
                        successful_tests += 1
                    
                    if isinstance(res, dict):
                        if 'avg_response_time_ms' in res:
                            response_times.append(res['avg_response_time_ms'])
                        if 'error_rate' in res:
                            error_rates.append(res['error_rate'])
                        if 'requests_per_second' in res:
                            throughputs.append(res['requests_per_second'])
                        if 'total_requests' in res:
                            overall_metrics['total_requests'] += res['total_requests']
                        
                else:
                    # Direct format
                    if result.get('error_rate', 100) < 10:
                        successful_tests += 1
                    
                    if 'average_response_time' in result:
                        response_times.append(result['average_response_time'])
                    if 'error_rate' in result:
                        error_rates.append(result['error_rate'])
                    if 'requests_per_second' in result:
                        throughputs.append(result['requests_per_second'])
                    if 'total_requests' in result:
                        overall_metrics['total_requests'] += result['total_requests']
            
            elif test_type == 'integration_test':
                # Check integration test success
                if isinstance(result, dict):
                    all_passed = all(r.get('success', False) for r in result.values() if isinstance(r, dict))
                    if all_passed:
                        successful_tests += 1
            
            elif test_type == 'frontend_test':
                # Check frontend performance
                overall = result.get('overall_metrics', {})
                if overall.get('avg_load_time', 10000) < 5000:  # Less than 5 seconds
                    successful_tests += 1
        
        # Calculate summary metrics
        if response_times:
            overall_metrics['avg_response_time'] = statistics.mean(response_times)
        if throughputs:
            overall_metrics['max_throughput'] = max(throughputs)
        if error_rates:
            overall_metrics['min_error_rate'] = min(error_rates)
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        html_content += f"""
                    <h3>Test Execution Summary</h3>
                    <p><strong>Tests Executed:</strong> {total_tests}</p>
                    <p><strong>Tests Passed:</strong> {successful_tests} ({success_rate:.1f}%)</p>
                    <p><strong>Overall Status:</strong> 
                        <span class="status-{'good' if success_rate >= 80 else 'warning' if success_rate >= 50 else 'error'}">
                            {'✅ EXCELLENT' if success_rate >= 90 else '⚠️ NEEDS ATTENTION' if success_rate >= 50 else '❌ CRITICAL ISSUES'}
                        </span>
                    </p>
                </div>
                
                <div class="metrics-grid">
"""
        
        # Add metric cards
        if overall_metrics['avg_response_time'] > 0:
            status = 'good' if overall_metrics['avg_response_time'] < 1000 else 'warning' if overall_metrics['avg_response_time'] < 3000 else 'error'
            html_content += f"""
                    <div class="metric-card status-{status}">
                        <h3>Average Response Time</h3>
                        <div class="metric-value">{overall_metrics['avg_response_time']:.0f}<span class="metric-unit">ms</span></div>
                    </div>
"""
        
        if overall_metrics['max_throughput'] > 0:
            status = 'good' if overall_metrics['max_throughput'] > 20 else 'warning' if overall_metrics['max_throughput'] > 10 else 'error'
            html_content += f"""
                    <div class="metric-card status-{status}">
                        <h3>Peak Throughput</h3>
                        <div class="metric-value">{overall_metrics['max_throughput']:.1f}<span class="metric-unit">req/s</span></div>
                    </div>
"""
        
        if overall_metrics['min_error_rate'] < 100:
            status = 'good' if overall_metrics['min_error_rate'] < 1 else 'warning' if overall_metrics['min_error_rate'] < 5 else 'error'
            html_content += f"""
                    <div class="metric-card status-{status}">
                        <h3>Best Error Rate</h3>
                        <div class="metric-value">{overall_metrics['min_error_rate']:.2f}<span class="metric-unit">%</span></div>
                    </div>
"""
        
        if overall_metrics['total_requests'] > 0:
            html_content += f"""
                    <div class="metric-card">
                        <h3>Total Requests</h3>
                        <div class="metric-value">{overall_metrics['total_requests']:,}<span class="metric-unit">requests</span></div>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
"""
        
        # Performance Charts
        if chart_files:
            html_content += """
            <div class="section">
                <h2>📈 Performance Charts</h2>
"""
            for chart_file in chart_files:
                chart_name = os.path.basename(chart_file).replace('_', ' ').replace('.png', '').title()
                html_content += f"""
                <div class="chart">
                    <h3>{chart_name}</h3>
                    <img src="{os.path.basename(chart_file)}" alt="{chart_name}">
                </div>
"""
            html_content += """
            </div>
"""
        
        # Detailed Test Results
        html_content += """
            <div class="section">
                <h2>🔍 Detailed Test Results</h2>
"""
        
        for test_type, result in results.items():
            if not result:
                continue
                
            test_name = test_type.replace('_', ' ').title()
            html_content += f"""
                <div class="test-summary">
                    <h3>{test_name}</h3>
"""
            
            if test_type in ['load_test', 'stress_test', 'endurance_test']:
                # Performance test results
                if 'results' in result:
                    res = result['results']
                    # Handle list/dict result structure
                    if isinstance(res, list) and len(res) > 0:
                        res = res[0]  # Use first scenario
                    
                    if isinstance(res, dict):
                        html_content += f"""
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            <tr><td>Total Requests</td><td>{res.get('total_requests', 0):,}</td></tr>
                            <tr><td>Successful Requests</td><td>{res.get('successful_requests', 0):,}</td></tr>
                            <tr><td>Failed Requests</td><td>{res.get('failed_requests', 0):,}</td></tr>
                            <tr><td>Success Rate</td><td>{res.get('success_rate', 0):.2f}%</td></tr>
                            <tr><td>Average Response Time</td><td>{res.get('avg_response_time_ms', 0):.2f}ms</td></tr>
                            <tr><td>Requests per Second</td><td>{res.get('requests_per_second', 0):.2f}</td></tr>
                        </table>
"""
                    else:
                        html_content += "<p>No detailed metrics available</p>"
                
            elif test_type == 'integration_test':
                # Integration test results
                total_scenarios = len(result)
                passed_scenarios = sum(1 for r in result.values() if isinstance(r, dict) and r.get('success', False))
                
                html_content += f"""
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Scenarios</td><td>{total_scenarios}</td></tr>
                        <tr><td>Passed Scenarios</td><td>{passed_scenarios}</td></tr>
                        <tr><td>Success Rate</td><td>{(passed_scenarios/total_scenarios*100) if total_scenarios > 0 else 0:.1f}%</td></tr>
                    </table>
"""
                
            elif test_type == 'frontend_test':
                # Frontend test results
                overall = result.get('overall_metrics', {})
                html_content += f"""
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Pages Tested</td><td>{overall.get('total_pages_tested', 0)}</td></tr>
                        <tr><td>Average Load Time</td><td>{overall.get('avg_load_time', 0):.0f}ms</td></tr>
                        <tr><td>Total Bundle Size</td><td>{overall.get('total_bundle_size', 0)/1024:.1f}KB</td></tr>
                        <tr><td>Average API Response Time</td><td>{overall.get('avg_api_response_time', 0):.0f}ms</td></tr>
                    </table>
"""
            
            html_content += """
                </div>
"""
        
        html_content += """
            </div>
"""
        
        # Recommendations
        recommendations = self.generate_recommendations(results)
        if recommendations:
            html_content += f"""
            <div class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendations">
                    <h3>Performance Optimization Suggestions</h3>
                    <ul>
"""
            for rec in recommendations:
                html_content += f"<li>{rec}</li>"
            
            html_content += """
                    </ul>
                </div>
            </div>
"""
        
        # Footer
        html_content += f"""
        </div>
        
        <div class="footer">
            <p>AI Istanbul Load Testing Suite | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Report includes Load Testing, Stress Testing, Endurance Testing, Integration Testing, and Frontend Performance Analysis</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_file = os.path.join(self.output_dir, 'load_testing_report.html')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_file
    
    def generate_enhanced_mobile_report(self, results: Dict[str, Any]) -> str:
        """Generate enhanced mobile location testing report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Mobile Location Testing Report - AI Istanbul</title>
            <style>
                {self.get_enhanced_css()}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Enhanced Mobile Location Testing Report</h1>
                    <div class="meta-info">
                        <p><strong>AI Istanbul Project</strong></p>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p>Test Duration: {results.get('summary', {}).get('total_test_time_seconds', 0)} seconds</p>
                    </div>
                </div>

                {self._generate_enhanced_summary_section(results)}
                {self._generate_gps_testing_section(results)}
                {self._generate_mobile_ux_section(results)}
                {self._generate_chat_scenarios_section(results)}
                {self._generate_performance_section(results)}
                {self._generate_accessibility_section(results)}
                {self._generate_recommendations_section(results)}
                
                <div class="footer">
                    <p>AI Istanbul Enhanced Mobile Location Testing Suite v2.0</p>
                    <p>For detailed analysis and improvements, see recommendations section above.</p>
                </div>
            </div>
            
            <script>
                {self.get_enhanced_javascript()}
            </script>
        </body>
        </html>
        """
        
        return html_content

    def _generate_enhanced_summary_section(self, results: Dict[str, Any]) -> str:
        """Generate enhanced summary section"""
        
        summary = results.get('summary', {})
        
        return f"""
        <div class="section">
            <h2>📊 Test Summary</h2>
            <div class="summary-grid">
                <div class="summary-card success">
                    <h3>{summary.get('overall_success_rate', 0)}%</h3>
                    <p>Overall Success Rate</p>
                </div>
                <div class="summary-card info">
                    <h3>{summary.get('total_locations_tested', 0)}</h3>
                    <p>Locations Tested</p>
                </div>
                <div class="summary-card info">
                    <h3>{summary.get('total_devices_tested', 0)}</h3>
                    <p>Devices Tested</p>
                </div>
                <div class="summary-card {'warning' if summary.get('total_errors', 0) > 0 else 'success'}">
                    <h3>{summary.get('total_errors', 0)}</h3>
                    <p>Total Errors</p>
                </div>
            </div>
        </div>
        """

    def _generate_gps_testing_section(self, results: Dict[str, Any]) -> str:
        """Generate GPS testing section"""
        
        gps_results = results.get('real_world_gps', {})
        location_tests = gps_results.get('location_accuracy_tests', {})
        edge_cases = gps_results.get('edge_case_handling', {})
        distance_tests = gps_results.get('distance_calculations', {})
        
        html = """
        <div class="section">
            <h2>🛰️ Real-World GPS Testing</h2>
            
            <div class="subsection">
                <h3>📍 Istanbul Location Accuracy</h3>
                <div class="location-grid">
        """
        
        for location_name, test_data in location_tests.items():
            success = test_data.get('validation_success', False)
            response_time = test_data.get('response_time_ms', 0)
            district = test_data.get('district', 'Unknown')
            
            status_class = 'success' if success else 'error'
            html += f"""
                <div class="location-card {status_class}">
                    <h4>{location_name}</h4>
                    <p><strong>District:</strong> {district}</p>
                    <p><strong>Status:</strong> {'✅ Valid' if success else '❌ Failed'}</p>
                    <p><strong>Response:</strong> {response_time:.1f}ms</p>
                    <p><strong>Nearby:</strong> {test_data.get('nearby_recommendations', 0)} attractions</p>
                </div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="subsection">
                <h3>⚠️ Edge Case Handling</h3>
                <div class="edge-case-list">
        """
        
        for case_name, case_data in edge_cases.items():
            handled = case_data.get('handled_correctly', False)
            expected = case_data.get('expected_error', 'Unknown')
            
            html += f"""
                <div class="edge-case {'success' if handled else 'error'}">
                    <strong>{case_name}</strong>
                    <span class="status">{'✅ Handled' if handled else '❌ Not Handled'}</span>
                    <small>Expected: {expected}</small>
                </div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="subsection">
                <h3>📏 Distance Calculation Accuracy</h3>
                <div class="distance-tests">
        """
        
        for route_name, dist_data in distance_tests.items():
            accuracy = dist_data.get('accuracy_percentage', 0)
            calculated = dist_data.get('calculated_distance_km', 0)
            api_distance = dist_data.get('api_distance_km', 0)
            
            accuracy_class = 'success' if accuracy > 95 else 'warning' if accuracy > 90 else 'error'
            
            html += f"""
                <div class="distance-test {accuracy_class}">
                    <strong>{route_name.replace('_', ' → ')}</strong>
                    <div class="distance-details">
                        <span>Calculated: {calculated}km</span>
                        <span>API: {api_distance}km</span>
                        <span>Accuracy: {accuracy}%</span>
                    </div>
                </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html

    def _generate_mobile_ux_section(self, results: Dict[str, Any]) -> str:
        """Generate mobile UX testing section"""
        
        mobile_results = results.get('enhanced_mobile_ux', {})
        device_performance = mobile_results.get('device_performance', {})
        touch_interactions = mobile_results.get('touch_interactions', {})
        offline_behavior = mobile_results.get('offline_behavior', {})
        
        html = """
        <div class="section">
            <h2>📱 Enhanced Mobile UX Testing</h2>
            
            <div class="subsection">
                <h3>⚡ Device Performance</h3>
                <div class="device-grid">
        """
        
        for device_name, perf_data in device_performance.items():
            load_time = perf_data.get('load_time_ms', 0)
            memory_usage = perf_data.get('memory_usage_mb', 0)
            performance_score = perf_data.get('performance_score', 'unknown')
            
            perf_class = 'success' if performance_score == 'good' else 'warning'
            
            html += f"""
                <div class="device-card {perf_class}">
                    <h4>{device_name}</h4>
                    <p><strong>Load Time:</strong> {load_time:.1f}ms</p>
                    <p><strong>Memory:</strong> {memory_usage:.1f}MB</p>
                    <p><strong>Viewport:</strong> {perf_data.get('viewport', 'Unknown')}</p>
                    <div class="score-badge {perf_class}">{performance_score}</div>
                </div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="subsection">
                <h3>👆 Touch Interactions</h3>
                <div class="touch-results">
        """
        
        for device_name, touch_data in touch_interactions.items():
            html += f"""<div class="touch-device">
                <h4>{device_name}</h4>
                <div class="touch-tests">
            """
            
            touch_tests = [
                ('tap_accuracy', 'Tap Accuracy'),
                ('swipe_gestures', 'Swipe Gestures'),
                ('touch_targets_adequate', 'Touch Targets'),
                ('long_press', 'Long Press')
            ]
            
            for test_key, test_name in touch_tests:
                passed = touch_data.get(test_key, False)
                status_class = 'success' if passed else 'warning'
                status_text = '✅ Pass' if passed else '⚠️ Needs Work'
                
                html += f"""
                    <div class="touch-test {status_class}">
                        <span>{test_name}</span>
                        <span class="status">{status_text}</span>
                    </div>
                """
            
            html += "</div></div>"
        
        html += """
                </div>
            </div>
            
            <div class="subsection">
                <h3>🌐 Offline Behavior</h3>
                <div class="offline-results">
        """
        
        offline_tests = [
            ('offline_page_loads', 'Page Loads Offline'),
            ('cache_effectiveness', 'Cache Effectiveness'),
            ('offline_message_shown', 'Offline Message'),
            ('graceful_degradation', 'Graceful Degradation')
        ]
        
        for test_key, test_name in offline_tests:
            passed = offline_behavior.get(test_key, False)
            status_class = 'success' if passed else 'error'
            status_text = '✅ Working' if passed else '❌ Failed'
            
            html += f"""
                <div class="offline-test {status_class}">
                    <span>{test_name}</span>
                    <span class="status">{status_text}</span>
                </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html

    def _generate_chat_scenarios_section(self, results: Dict[str, Any]) -> str:
        """Generate location-based chat scenarios section"""
        
        chat_results = results.get('location_based_chat', {})
        location_responses = chat_results.get('location_specific_responses', {})
        
        html = """
        <div class="section">
            <h2>💬 Location-Based Chat Scenarios</h2>
        """
        
        for location_name, scenario_data in location_responses.items():
            location = scenario_data.get('location', {})
            queries = scenario_data.get('query_responses', [])
            context_maintained = scenario_data.get('context_maintained', False)
            recommendations_relevant = scenario_data.get('recommendations_relevant', False)
            
            html += f"""
            <div class="subsection">
                <h3>📍 {location_name}</h3>
                <div class="location-info">
                    <p><strong>Coordinates:</strong> {location.get('lat', 0):.4f}, {location.get('lng', 0):.4f}</p>
                    <p><strong>Context Maintained:</strong> {'✅ Yes' if context_maintained else '❌ No'}</p>
                    <p><strong>Relevant Recommendations:</strong> {'✅ Yes' if recommendations_relevant else '❌ No'}</p>
                </div>
                
                <div class="chat-queries">
            """
            
            for query_data in queries:
                query = query_data.get('query', '')
                response_length = query_data.get('response_length', 0)
                location_context = query_data.get('location_context', False)
                specific_info = query_data.get('specific_info', False)
                preview = query_data.get('response_preview', '')
                
                context_class = 'success' if location_context else 'warning'
                info_class = 'success' if specific_info else 'warning'
                
                html += f"""
                    <div class="chat-query">
                        <div class="query"><strong>Q:</strong> {query}</div>
                        <div class="response-info">
                            <span class="length">Length: {response_length} chars</span>
                            <span class="context {context_class}">Context: {'✅' if location_context else '❌'}</span>
                            <span class="specific {info_class}">Specific: {'✅' if specific_info else '❌'}</span>
                        </div>
                        <div class="response-preview">{preview}</div>
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        html += "</div>"
        return html

    def _generate_performance_section(self, results: Dict[str, Any]) -> str:
        """Generate performance under load section"""
        
        perf_results = results.get('performance_under_load', {})
        concurrent_requests = perf_results.get('concurrent_requests', {})
        memory_leaks = perf_results.get('memory_leaks', {})
        network_conditions = perf_results.get('network_conditions', {})
        
        html = """
        <div class="section">
            <h2>⚡ Performance Under Load</h2>
            
            <div class="subsection">
                <h3>🔄 Concurrent Request Testing</h3>
                <div class="concurrent-stats">
        """
        
        total_requests = concurrent_requests.get('total_requests', 0)
        successful_requests = concurrent_requests.get('successful_requests', 0)
        success_rate = concurrent_requests.get('success_rate', 0)
        avg_response_time = concurrent_requests.get('average_response_time', 0)
        
        html += f"""
                    <div class="stat-card">
                        <h4>Total Requests</h4>
                        <p class="stat-value">{total_requests}</p>
                    </div>
                    <div class="stat-card">
                        <h4>Success Rate</h4>
                        <p class="stat-value {'success' if success_rate > 90 else 'warning'}">{success_rate:.1f}%</p>
                    </div>
                    <div class="stat-card">
                        <h4>Avg Response Time</h4>
                        <p class="stat-value">{avg_response_time:.2f}s</p>
                    </div>
                </div>
            </div>
            
            <div class="subsection">
                <h3>🧠 Memory Leak Detection</h3>
        """
        
        if memory_leaks:
            initial_memory = memory_leaks.get('initial_memory_mb', 0)
            final_memory = memory_leaks.get('final_memory_mb', 0)
            memory_trend = memory_leaks.get('memory_trend', 'unknown')
            potential_leak = memory_leaks.get('potential_leak', False)
            
            trend_class = 'error' if potential_leak else 'success'
            
            html += f"""
                <div class="memory-analysis {trend_class}">
                    <p><strong>Initial Memory:</strong> {initial_memory:.1f}MB</p>
                    <p><strong>Final Memory:</strong> {final_memory:.1f}MB</p>
                    <p><strong>Trend:</strong> {memory_trend}</p>
                    <p><strong>Potential Leak:</strong> {'⚠️ Yes' if potential_leak else '✅ No'}</p>
                </div>
            """
        else:
            html += "<p>Memory leak testing not available</p>"
        
        html += """
            </div>
            
            <div class="subsection">
                <h3>🌐 Network Conditions</h3>
                <div class="network-tests">
        """
        
        for condition_name, condition_data in network_conditions.items():
            load_time = condition_data.get('load_time_ms', 0)
            acceptable = condition_data.get('acceptable_performance', False)
            
            perf_class = 'success' if acceptable else 'warning'
            
            html += f"""
                <div class="network-test {perf_class}">
                    <h4>{condition_name}</h4>
                    <p>Load Time: {load_time:.1f}ms</p>
                    <p>Performance: {'✅ Acceptable' if acceptable else '⚠️ Slow'}</p>
                </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
        
        return html

    def _generate_accessibility_section(self, results: Dict[str, Any]) -> str:
        """Generate accessibility testing section"""
        
        mobile_results = results.get('enhanced_mobile_ux', {})
        accessibility_results = mobile_results.get('accessibility', {})
        
        html = """
        <div class="section">
            <h2>♿ Accessibility Testing</h2>
        """
        
        if accessibility_results:
            for device_name, access_data in accessibility_results.items():
                html += f"""
                <div class="subsection">
                    <h3>📱 {device_name}</h3>
                    <div class="accessibility-tests">
                """
                
                access_tests = [
                    ('proper_headings', 'Proper Headings'),
                    ('alt_text_present', 'Alt Text Present'),
                    ('keyboard_navigation', 'Keyboard Navigation'),
                    ('aria_labels', 'ARIA Labels')
                ]
                
                for test_key, test_name in access_tests:
                    passed = access_data.get(test_key, False)
                    status_class = 'success' if passed else 'error'
                    status_text = '✅ Pass' if passed else '❌ Fail'
                    
                    html += f"""
                        <div class="accessibility-test {status_class}">
                            <span>{test_name}</span>
                            <span class="status">{status_text}</span>
                        </div>
                    """
                
                html += """
                    </div>
                </div>
                """
        else:
            html += "<p>Accessibility testing results not available</p>"
        
        html += "</div>"
        return html

    def _generate_recommendations_section(self, results: Dict[str, Any]) -> str:
        """Generate recommendations section based on test results"""
        
        recommendations = []
        
        # Analyze results and generate recommendations
        gps_results = results.get('real_world_gps', {})
        mobile_results = results.get('enhanced_mobile_ux', {})
        chat_results = results.get('location_based_chat', {})
        perf_results = results.get('performance_under_load', {})
        
        # GPS recommendations
        location_tests = gps_results.get('location_accuracy_tests', {})
        failed_locations = [name for name, data in location_tests.items() if not data.get('validation_success', False)]
        
        if failed_locations:
            recommendations.append({
                'category': 'GPS Accuracy',
                'priority': 'High',
                'issue': f'Location validation failed for {len(failed_locations)} locations',
                'recommendation': 'Implement better error handling and fallback mechanisms for GPS validation',
                'locations': failed_locations
            })
        
        # Performance recommendations
        device_performance = mobile_results.get('device_performance', {})
        slow_devices = [name for name, data in device_performance.items() if data.get('performance_score') != 'good']
        
        if slow_devices:
            recommendations.append({
                'category': 'Mobile Performance',
                'priority': 'Medium',
                'issue': f'Performance issues on {len(slow_devices)} devices',
                'recommendation': 'Optimize bundle size, implement lazy loading, and improve caching strategies',
                'devices': slow_devices
            })
        
        # Accessibility recommendations
        accessibility_results = mobile_results.get('accessibility', {})
        accessibility_issues = []
        
        for device_name, access_data in accessibility_results.items():
            failed_tests = [test for test, passed in access_data.items() if not passed and test != 'errors']
            if failed_tests:
                accessibility_issues.extend(failed_tests)
        
        if accessibility_issues:
            recommendations.append({
                'category': 'Accessibility',
                'priority': 'High',
                'issue': f'Accessibility issues found: {set(accessibility_issues)}',
                'recommendation': 'Implement proper ARIA labels, semantic HTML, and keyboard navigation support'
            })
        
        # Memory leak recommendations
        memory_leaks = perf_results.get('memory_leaks', {})
        if memory_leaks.get('potential_leak', False):
            recommendations.append({
                'category': 'Memory Management',
                'priority': 'High',
                'issue': 'Potential memory leak detected',
                'recommendation': 'Review event listeners cleanup, implement proper component unmounting, and optimize DOM manipulation'
            })
        
        # Chat context recommendations
        location_responses = chat_results.get('location_specific_responses', {})
        context_issues = [name for name, data in location_responses.items() if not data.get('context_maintained', False)]
        
        if context_issues:
            recommendations.append({
                'category': 'Chat Context',
                'priority': 'Medium',
                'issue': f'Location context not maintained for {len(context_issues)} locations',
                'recommendation': 'Improve location context awareness in chat responses and implement better session state management'
            })
        
        html = """
        <div class="section">
            <h2>🎯 Recommendations & Action Items</h2>
        """
        
        if recommendations:
            for rec in recommendations:
                priority_class = rec['priority'].lower();
                
                html += f"""
                <div class="recommendation {priority_class}">
                    <div class="rec-header">
                        <h3>{rec['category']}</h3>
                        <span class="priority-badge {priority_class}">{rec['priority']} Priority</span>
                    </div>
                    <div class="rec-content">
                        <p><strong>Issue:</strong> {rec['issue']}</p>
                        <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                """
                
                if 'locations' in rec:
                    html += f"<p><strong>Affected Locations:</strong> {', '.join(rec['locations'])}</p>"
                if 'devices' in rec:
                    html += f"<p><strong>Affected Devices:</strong> {', '.join(rec['devices'])}</p>"
                
                html += """
                    </div>
                </div>
                """
        else:
            html += """
            <div class="no-recommendations">
                <h3>🎉 Excellent Results!</h3>
                <p>No critical issues found. The mobile location testing shows good performance across all categories.</p>
            </div>
            """
        
        html += "</div>"
        return html

    def get_enhanced_css(self) -> str:
        """Get enhanced CSS for mobile testing report"""
        
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0 0 20px 0;
            font-size: 2.5em;
            font-weight: 700;
        }
        
        .meta-info p {
            margin: 5px 0;
            opacity: 0.9;
        }
        
        .section {
            padding: 40px;
            border-bottom: 1px solid #eee;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .section h2 {
            color: #1a1a2e;
            margin-bottom: 30px;
            font-size: 1.8em;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }
        
        .subsection {
            margin-bottom: 30px;
        }
        
        .subsection h3 {
            color: #444;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        /* Summary Grid */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #ddd;
        }
        
        .summary-card.success { border-left-color: #22c55e; }
        .summary-card.warning { border-left-color: #f59e0b; }
        .summary-card.error { border-left-color: #ef4444; }
        .summary-card.info { border-left-color: #3b82f6; }
        
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 700;
        }
        
        .summary-card.success h3 { color: #22c55e; }
        .summary-card.warning h3 { color: #f59e0b; }
        .summary-card.error h3 { color: #ef4444; }
        .summary-card.info h3 { color: #3b82f6; }
        
        /* Location Grid */
        .location-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .location-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #ddd;
        }
        
        .location-card.success { border-left-color: #22c55e; }
        .location-card.error { border-left-color: #ef4444; }
        
        .location-card h4 {
            margin: 0 0 15px 0;
            color: #1a1a2e;
            font-size: 1.1em;
        }
        
        .location-card p {
            margin: 8px 0;
            font-size: 0.9em;
        }
        
        /* Device Grid */
        .device-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
        }
        
        .device-card {
            position: relative;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #ddd;
        }
        
        .device-card.success { border-left-color: #22c55e; }
        .device-card.warning { border-left-color: #f59e0b; }
        
        .score-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .score-badge.success {
            background: #dcfce7;
            color: #166534;
        }
        
        .score-badge.warning {
            background: #fef3c7;
            color: #92400e;
        }
        
        /* Touch Results */
        .touch-results {
            display: grid;
            gap: 20px;
        }
        
        .touch-device {
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
        }
        
        .touch-tests {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .touch-test {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            background: white;
            border-radius: 6px;
            border-left: 3px solid #ddd;
        }
        
        .touch-test.success { border-left-color: #22c55e; }
        .touch-test.warning { border-left-color: #f59e0b; }
        
        /* Chat Queries */
        .chat-queries {
            display: grid;
            gap: 15px;
        }
        
        .chat-query {
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }
        
        .query {
            font-weight: 600;
            margin-bottom: 10px;
            color: #1a1a2e;
        }
        
        .response-info {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .response-info span.success { color: #22c55e; }
        .response-info span.warning { color: #f59e0b; }
        
        .response-preview {
            background: white;
            padding: 15px;
            border-radius: 6px;
            font-style: italic;
            border-left: 3px solid #e5e7eb;
        }
        
        /* Recommendations */
        .recommendation {
            background: white;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: #f8fafc;
            border-left: 4px solid #ddd;
        }
        
        .recommendation.high .rec-header { border-left-color: #ef4444; }
        .recommendation.medium .rec-header { border-left-color: #f59e0b; }
        .recommendation.low .rec-header { border-left-color: #3b82f6; }
        
        .priority-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .priority-badge.high {
            background: #fecaca;
            color: #991b1b;
        }
        
        .priority-badge.medium {
            background: #fef3c7;
            color: #92400e;
        }
        
        .priority-badge.low {
            background: #dbeafe;
            color: #1e40af;
        }
        
        .rec-content {
            padding: 20px;
        }
        
        .rec-content p {
            margin: 10px 0;
        }
        
        /* Statistics */
        .concurrent-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stat-card h4 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stat-value {
            margin: 0;
            font-size: 2em;
            font-weight: 700;
            color: #1a1a2e;
        }
        
        .stat-value.success { color: #22c55e; }
        .stat-value.warning { color: #f59e0b; }
        .stat-value.error { color: #ef4444; }
        
        /* Network Tests */
        .network-tests {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .network-test {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #ddd;
        }
        
        .network-test.success { border-left-color: #22c55e; }
        .network-test.warning { border-left-color: #f59e0b; }
        
        /* Footer */
        .footer {
            background: #1a1a2e;
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .footer p {
            margin: 5px 0;
            opacity: 0.8;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 8px;
            }
            
            .header, .section {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            .location-grid, .device-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def get_enhanced_javascript(self) -> str:
        """Get enhanced JavaScript for mobile testing report"""
        
        return """
        // Enhanced mobile testing report interactions
        document.addEventListener('DOMContentLoaded', function() {
            // Add click-to-expand functionality for detailed results
            const sections = document.querySelectorAll('.section');
            
            sections.forEach(section => {
                const header = section.querySelector('h2');
                if (header) {
                    header.style.cursor = 'pointer';
                    header.addEventListener('click', function() {
                        const content = section.querySelector('.subsection');
                        if (content) {
                            content.style.display = content.style.display === 'none' ? 'block' : 'none';
                        }
                    });
                }
            });
            
            // Add smooth scrolling for navigation
            const smoothScroll = (target) => {
                document.querySelector(target).scrollIntoView({
                    behavior: 'smooth'
                });
            };
            
            // Add print functionality
            const printBtn = document.createElement('button');
            printBtn.textContent = '🖨️ Print Report';
            printBtn.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                z-index: 1000;
            `;
            
            printBtn.addEventListener('click', () => window.print());
            document.body.appendChild(printBtn);
            
            // Add export functionality
            const exportBtn = document.createElement('button');
            exportBtn.textContent = '📊 Export Data';
            exportBtn.style.cssText = `
                position: fixed;
                top: 70px;
                right: 20px;
                background: #22c55e;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                z-index: 1000;
            `;
            
            exportBtn.addEventListener('click', function() {
                // Extract data for CSV export
                const data = extractReportData();
                downloadCSV(data, 'enhanced_mobile_test_results.csv');
            });
            
            document.body.appendChild(exportBtn);
            
            function extractReportData() {
                // Extract key metrics from the report
                const rows = [];
                
                // Add summary data
                const summaryCards = document.querySelectorAll('.summary-card');
                summaryCards.forEach(card => {
                    const metric = card.querySelector('p').textContent;
                    const value = card.querySelector('h3').textContent;
                    rows.push(['Summary', metric, value, '']);
                });
                
                // Add location data
                const locationCards = document.querySelectorAll('.location-card');
                locationCards.forEach(card => {
                    const name = card.querySelector('h4').textContent;
                    const district = card.textContent.match(/District: ([^\\n]+)/)?.[1] || '';
                    const status = card.textContent.match(/Status: ([^\\n]+)/)?.[1] || '';
                    const responseTime = card.textContent.match(/Response: ([^\\n]+)/)?.[1] || '';
                    rows.push(['Location', name, district, status, responseTime]);
                });
                
                return rows;
            }
            
            function downloadCSV(data, filename) {
                const csv = data.map(row => row.join(',')).join('\\n');
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            }
            
            // Add progress indicators for test results
            const progressBars = document.querySelectorAll('.stat-value');
            progressBars.forEach(bar => {
                const value = parseFloat(bar.textContent);
                if (value && value <= 100) {
                    // Add visual progress indicator
                    const progress = document.createElement('div');
                    progress.style.cssText = `
                        width: 100%;
                        height: 4px;
                        background: #e5e7eb;
                        border-radius: 2px;
                        margin-top: 8px;
                        overflow: hidden;
                    `;
                    
                    const fill = document.createElement('div');
                    fill.style.cssText = `
                        width: ${value}%;
                        height: 100%;
                        background: ${value > 90 ? '#22c55e' : value > 70 ? '#f59e0b' : '#ef4444'};
                        transition: width 0.3s ease;
                    `;
                    
                    progress.appendChild(fill);
                    bar.parentNode.appendChild(progress);
                }
            });
        });
        """
