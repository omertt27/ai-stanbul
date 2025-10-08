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
    
    def generate_recommendations(self, results: Dict[str, Dict]) -> List[str]:
        """Generate performance recommendations based on test results"""
        
        recommendations = []
        
        # Analyze response times
        response_times = []
        for test_type in ['load_test', 'stress_test', 'endurance_test']:
            if test_type in results and results[test_type]:
                result = results[test_type]
                if 'results' in result and 'avg_response_time_ms' in result['results']:
                    response_times.append(result['results']['avg_response_time_ms'])
                elif 'average_response_time' in result:
                    response_times.append(result['average_response_time'])
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            if avg_response_time > 2000:
                recommendations.append("Consider optimizing API response times - average exceeds 2 seconds")
            if avg_response_time > 5000:
                recommendations.append("Critical: Response times are very high (>5s) - immediate optimization needed")
        
        # Analyze error rates
        error_rates = []
        for test_type in ['load_test', 'stress_test', 'endurance_test']:
            if test_type in results and results[test_type]:
                result = results[test_type]
                if 'results' in result and 'error_rate' in result['results']:
                    error_rates.append(result['results']['error_rate'])
                elif 'error_rate' in result:
                    error_rates.append(result['error_rate'])
        
        if error_rates:
            max_error_rate = max(error_rates)
            if max_error_rate > 5:
                recommendations.append("High error rates detected - investigate and fix error sources")
            if max_error_rate > 10:
                recommendations.append("Critical: Error rates exceed 10% - system stability issues need immediate attention")
        
        # Analyze throughput
        throughputs = []
        for test_type in ['load_test', 'stress_test']:
            if test_type in results and results[test_type]:
                result = results[test_type]
                if 'results' in result and 'requests_per_second' in result['results']:
                    throughputs.append(result['results']['requests_per_second'])
                elif 'requests_per_second' in result:
                    throughputs.append(result['requests_per_second'])
        
        if throughputs:
            max_throughput = max(throughputs)
            if max_throughput < 20:
                recommendations.append("Low throughput detected - consider scaling or performance optimization")
        
        # Memory analysis for endurance test
        if 'endurance_test' in results and results['endurance_test']:
            memory_analysis = results['endurance_test'].get('memory_analysis', {})
            if memory_analysis.get('potential_leak', False):
                recommendations.append("Potential memory leak detected in endurance test - review memory usage patterns")
        
        # Frontend performance
        if 'frontend_test' in results and results['frontend_test']:
            overall = results['frontend_test'].get('overall_metrics', {})
            if overall.get('avg_load_time', 0) > 3000:
                recommendations.append("Frontend load times are high - optimize bundle size and loading strategy")
            if overall.get('total_bundle_size', 0) > 2000000:
                recommendations.append("Large bundle size detected - consider code splitting and lazy loading")
        
        # Integration test failures
        if 'integration_test' in results and results['integration_test']:
            failed_scenarios = [name for name, result in results['integration_test'].items() 
                              if isinstance(result, dict) and not result.get('success', True)]
            if failed_scenarios:
                recommendations.append(f"Integration test failures in: {', '.join(failed_scenarios)} - fix these workflows before production")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance looks good! Consider implementing continuous performance monitoring")
        
        recommendations.append("Set up automated performance testing in CI/CD pipeline")
        recommendations.append("Monitor these metrics in production with alerting thresholds")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report from all available test results"""
        
        console.print(f"\n[bold blue]📊 Generating Comprehensive Load Testing Report[/bold blue]")
        
        # Find latest results
        result_files = self.find_latest_results()
        console.print(f"[cyan]Found {len([f for f in result_files.values() if f])} result files[/cyan]")
        
        # Load results
        results = self.load_test_results(result_files)
        
        if not results:
            console.print("[red]❌ No test results found. Run some tests first![/red]")
            return ""
        
        # Generate charts
        console.print(f"[yellow]📈 Generating performance charts...[/yellow]")
        chart_files = self.generate_performance_charts(results)
        console.print(f"[green]✅ Generated {len(chart_files)} charts[/green]")
        
        # Generate HTML report
        console.print(f"[yellow]📝 Generating HTML report...[/yellow]")
        report_file = self.generate_html_report(results, chart_files)
        
        console.print(f"[bold green]✅ Comprehensive report generated: {report_file}[/bold green]")
        console.print(f"[cyan]📁 Report directory: {os.path.abspath(self.output_dir)}[/cyan]")
        
        return report_file

def main():
    """Main report generation execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Istanbul Load Testing Report Generator')
    parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    console.print(f"[bold]📊 AI Istanbul Load Testing Report Generator[/bold]")
    
    # Generate report
    generator = ReportGenerator(args.output_dir)
    report_file = generator.generate_comprehensive_report()
    
    if report_file:
        console.print(f"\n[bold green]🎉 Report generation complete![/bold green]")
        console.print(f"[cyan]Open the report: file://{os.path.abspath(report_file)}[/cyan]")
    else:
        console.print(f"\n[red]❌ Report generation failed[/red]")

if __name__ == "__main__":
    main()
