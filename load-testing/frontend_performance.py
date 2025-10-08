"""
AI Istanbul Frontend Performance Testing
Test frontend loading times, bundle sizes, and user interactions
"""

import time
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from config import get_config

console = Console()

@dataclass
class PerformanceMetrics:
    """Frontend performance metrics"""
    page_load_time: float
    first_contentful_paint: float
    largest_contentful_paint: float
    cumulative_layout_shift: float
    first_input_delay: float
    bundle_sizes: Dict[str, int]
    api_response_times: List[float]
    lighthouse_score: Optional[Dict] = None

class FrontendPerformanceTester:
    """Frontend performance testing with Playwright"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.browser = None
        self.context = None
        
    async def setup_browser(self, headless: bool = True):
        """Setup browser for testing"""
        playwright = await async_playwright().start()
        
        # Launch browser with performance monitoring enabled
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--disable-dev-shm-usage',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-web-security',
                '--disable-features=TranslateUI',
                '--disable-component-extensions-with-background-pages',
                '--disable-default-apps',
                '--disable-background-networking',
                '--disable-sync',
                '--metrics-recording-only',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--disable-ipc-flooding-protection'
            ]
        )
        
        # Create context with performance tracking
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Enable request interception for API monitoring
        await self.context.route('**/*', self._intercept_requests)
        
    async def _intercept_requests(self, route, request):
        """Intercept and monitor API requests"""
        # Continue with the request
        response = await route.fetch()
        
        # Log API requests
        if '/api/' in request.url or 'localhost:8000' in request.url:
            console.print(f"[dim]API Request: {request.method} {request.url} -> {response.status}[/dim]")
        
        await route.fulfill(response=response)
    
    async def cleanup(self):
        """Cleanup browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def measure_page_performance(self, page_path: str = '/') -> PerformanceMetrics:
        """Measure comprehensive page performance"""
        
        page = await self.context.new_page()
        
        # Enable performance monitoring
        await page.add_init_script("""
            window.performanceData = {
                navigationStart: 0,
                domContentLoaded: 0,
                loadComplete: 0,
                firstPaint: 0,
                firstContentfulPaint: 0,
                apiRequests: []
            };
            
            // Monitor navigation timing
            window.addEventListener('load', () => {
                const navigation = performance.getEntriesByType('navigation')[0];
                window.performanceData.navigationStart = navigation.startTime;
                window.performanceData.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.startTime;
                window.performanceData.loadComplete = navigation.loadEventEnd - navigation.startTime;
            });
            
            // Monitor paint timing
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.name === 'first-paint') {
                        window.performanceData.firstPaint = entry.startTime;
                    }
                    if (entry.name === 'first-contentful-paint') {
                        window.performanceData.firstContentfulPaint = entry.startTime;
                    }
                }
            });
            observer.observe({entryTypes: ['paint']});
            
            // Monitor API requests
            const originalFetch = window.fetch;
            window.fetch = function(...args) {
                const startTime = performance.now();
                return originalFetch.apply(this, args).then(response => {
                    const endTime = performance.now();
                    if (args[0].includes('/api/') || args[0].includes('localhost:8000')) {
                        window.performanceData.apiRequests.push({
                            url: args[0],
                            duration: endTime - startTime,
                            status: response.status
                        });
                    }
                    return response;
                });
            };
        """)
        
        try:
            # Navigate to page and measure load time
            start_time = time.time()
            await page.goto(f"{self.base_url}{page_path}", wait_until='networkidle')
            page_load_time = (time.time() - start_time) * 1000
            
            # Wait for page to be fully interactive
            await page.wait_for_load_state('domcontentloaded')
            await page.wait_for_timeout(2000)  # Allow for async operations
            
            # Get Core Web Vitals
            web_vitals = await page.evaluate("""
                () => {
                    return new Promise((resolve) => {
                        let lcp = 0;
                        let cls = 0;
                        let fid = 0;
                        
                        // Largest Contentful Paint
                        new PerformanceObserver((entryList) => {
                            const entries = entryList.getEntries();
                            const lastEntry = entries[entries.length - 1];
                            lcp = lastEntry.startTime;
                        }).observe({entryTypes: ['largest-contentful-paint']});
                        
                        // Cumulative Layout Shift
                        new PerformanceObserver((entryList) => {
                            for (const entry of entryList.getEntries()) {
                                if (!entry.hadRecentInput) {
                                    cls += entry.value;
                                }
                            }
                        }).observe({entryTypes: ['layout-shift']});
                        
                        // First Input Delay
                        new PerformanceObserver((entryList) => {
                            const firstInput = entryList.getEntries()[0];
                            if (firstInput) {
                                fid = firstInput.processingStart - firstInput.startTime;
                            }
                        }).observe({entryTypes: ['first-input']});
                        
                        // Get performance data after a delay
                        setTimeout(() => {
                            resolve({
                                lcp,
                                cls,
                                fid,
                                performanceData: window.performanceData
                            });
                        }, 3000);
                    });
                }
            """)
            
            # Get resource sizes (bundle analysis)
            resource_sizes = await page.evaluate("""
                () => {
                    const resources = performance.getEntriesByType('resource');
                    const sizes = {};
                    
                    resources.forEach(resource => {
                        if (resource.name.includes('.js') || resource.name.includes('.css')) {
                            const name = resource.name.split('/').pop();
                            sizes[name] = resource.transferSize || resource.encodedBodySize || 0;
                        }
                    });
                    
                    return sizes;
                }
            """)
            
            # Extract API response times
            api_response_times = []
            perf_data = web_vitals.get('performanceData', {})
            api_requests = perf_data.get('apiRequests', [])
            
            for request in api_requests:
                api_response_times.append(request['duration'])
            
            return PerformanceMetrics(
                page_load_time=page_load_time,
                first_contentful_paint=perf_data.get('firstContentfulPaint', 0),
                largest_contentful_paint=web_vitals.get('lcp', 0),
                cumulative_layout_shift=web_vitals.get('cls', 0),
                first_input_delay=web_vitals.get('fid', 0),
                bundle_sizes=resource_sizes,
                api_response_times=api_response_times
            )
            
        finally:
            await page.close()
    
    async def test_user_interactions(self, page_path: str = '/') -> Dict[str, float]:
        """Test user interaction performance"""
        
        page = await self.context.new_page()
        interaction_times = {}
        
        try:
            await page.goto(f"{self.base_url}{page_path}", wait_until='networkidle')
            
            # Test chat interaction if on main page
            if page_path == '/':
                try:
                    # Test search bar interaction
                    search_input = page.locator('input[placeholder*="Ask"], input[placeholder*="search"], textarea')
                    if await search_input.count() > 0:
                        start_time = time.time()
                        await search_input.first.click()
                        await search_input.first.fill("test query")
                        interaction_times['search_input'] = (time.time() - start_time) * 1000
                        
                        # Test form submission
                        submit_button = page.locator('button[type="submit"], button:has-text("Send"), button:has-text("Search")')
                        if await submit_button.count() > 0:
                            start_time = time.time()
                            await submit_button.first.click()
                            await page.wait_for_timeout(1000)  # Wait for response
                            interaction_times['form_submit'] = (time.time() - start_time) * 1000
                            
                except Exception as e:
                    console.print(f"[yellow]Chat interaction test failed: {e}[/yellow]")
            
            # Test navigation
            nav_links = page.locator('nav a, header a')
            if await nav_links.count() > 0:
                try:
                    start_time = time.time()
                    await nav_links.first.click()
                    await page.wait_for_load_state('networkidle')
                    interaction_times['navigation'] = (time.time() - start_time) * 1000
                except Exception as e:
                    console.print(f"[yellow]Navigation test failed: {e}[/yellow]")
            
            # Test scroll performance
            try:
                start_time = time.time()
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(500)
                await page.evaluate("window.scrollTo(0, 0)")
                interaction_times['scroll_performance'] = (time.time() - start_time) * 1000
            except Exception as e:
                console.print(f"[yellow]Scroll test failed: {e}[/yellow]")
            
            return interaction_times
            
        finally:
            await page.close()
    
    async def test_mobile_performance(self, page_path: str = '/') -> PerformanceMetrics:
        """Test mobile performance"""
        
        # Create mobile context
        mobile_context = await self.browser.new_context(
            viewport={'width': 375, 'height': 667},  # iPhone SE
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            is_mobile=True,
            has_touch=True
        )
        
        page = await mobile_context.new_page()
        
        try:
            # Add mobile-specific performance monitoring
            await page.add_init_script("""
                window.mobilePerformanceData = {
                    touchStartTime: 0,
                    touchEndTime: 0,
                    scrollStartTime: 0,
                    scrollEndTime: 0
                };
                
                document.addEventListener('touchstart', () => {
                    window.mobilePerformanceData.touchStartTime = performance.now();
                });
                
                document.addEventListener('touchend', () => {
                    window.mobilePerformanceData.touchEndTime = performance.now();
                });
            """)
            
            start_time = time.time()
            await page.goto(f"{self.base_url}{page_path}", wait_until='networkidle')
            mobile_load_time = (time.time() - start_time) * 1000
            
            # Test touch interactions
            try:
                # Test tap
                await page.tap('body')
                await page.wait_for_timeout(100)
                
                # Test scroll
                await page.evaluate("window.scrollBy(0, 100)")
                await page.wait_for_timeout(100)
                
            except Exception as e:
                console.print(f"[yellow]Mobile interaction test failed: {e}[/yellow]")
            
            # Get mobile-specific metrics
            mobile_data = await page.evaluate("window.mobilePerformanceData")
            
            return PerformanceMetrics(
                page_load_time=mobile_load_time,
                first_contentful_paint=0,  # Simplified for mobile
                largest_contentful_paint=0,
                cumulative_layout_shift=0,
                first_input_delay=0,
                bundle_sizes={},
                api_response_times=[]
            )
            
        finally:
            await page.close()
            await mobile_context.close()
    
    async def run_comprehensive_frontend_test(self) -> Dict[str, Any]:
        """Run comprehensive frontend performance test"""
        
        console.print(f"\n[bold blue]🎨 Frontend Performance Testing[/bold blue]")
        console.print(f"[cyan]Testing URL: {self.base_url}[/cyan]")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'pages_tested': [],
            'overall_metrics': {},
            'interaction_tests': {},
            'mobile_performance': {}
        }
        
        # Pages to test
        pages_to_test = [
            ('/', 'Home Page'),
            ('/chat', 'Chat Page'),
            ('/blog', 'Blog Page'),
            ('/about', 'About Page')
        ]
        
        with Progress() as progress:
            task = progress.add_task("Testing frontend performance...", total=len(pages_to_test) * 3)
            
            for page_path, page_name in pages_to_test:
                try:
                    console.print(f"[yellow]Testing {page_name}...[/yellow]")
                    
                    # Desktop performance
                    progress.update(task, description=f"Testing {page_name} - Desktop")
                    desktop_metrics = await self.measure_page_performance(page_path)
                    progress.advance(task)
                    
                    # User interactions
                    progress.update(task, description=f"Testing {page_name} - Interactions")
                    interaction_metrics = await self.test_user_interactions(page_path)
                    progress.advance(task)
                    
                    # Mobile performance
                    progress.update(task, description=f"Testing {page_name} - Mobile")
                    mobile_metrics = await self.test_mobile_performance(page_path)
                    progress.advance(task)
                    
                    # Store results
                    page_results = {
                        'path': page_path,
                        'name': page_name,
                        'desktop_metrics': desktop_metrics.__dict__,
                        'interaction_metrics': interaction_metrics,
                        'mobile_metrics': mobile_metrics.__dict__
                    }
                    
                    results['pages_tested'].append(page_results)
                    
                except Exception as e:
                    console.print(f"[red]Error testing {page_name}: {e}[/red]")
                    progress.advance(task, 3)  # Skip all three sub-tests
        
        # Calculate overall metrics
        if results['pages_tested']:
            all_load_times = []
            all_bundle_sizes = {}
            all_api_times = []
            
            for page_result in results['pages_tested']:
                desktop = page_result['desktop_metrics']
                all_load_times.append(desktop['page_load_time'])
                
                # Aggregate bundle sizes
                for bundle, size in desktop['bundle_sizes'].items():
                    if bundle not in all_bundle_sizes:
                        all_bundle_sizes[bundle] = []
                    all_bundle_sizes[bundle].append(size)
                
                # Aggregate API times
                all_api_times.extend(desktop['api_response_times'])
            
            results['overall_metrics'] = {
                'avg_load_time': statistics.mean(all_load_times) if all_load_times else 0,
                'max_load_time': max(all_load_times) if all_load_times else 0,
                'total_bundle_size': sum(max(sizes) for sizes in all_bundle_sizes.values()),
                'avg_api_response_time': statistics.mean(all_api_times) if all_api_times else 0,
                'total_pages_tested': len(results['pages_tested'])
            }
        
        return results
    
    def generate_frontend_report(self, results: Dict[str, Any]):
        """Generate frontend performance report"""
        
        console.print(f"\n[bold green]🎨 Frontend Performance Results[/bold green]")
        
        overall = results.get('overall_metrics', {})
        pages_tested = results.get('pages_tested', [])
        
        # Overall summary
        summary_table = Table(title="📊 Overall Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        summary_table.add_column("Status", style="green")
        
        avg_load_time = overall.get('avg_load_time', 0)
        max_load_time = overall.get('max_load_time', 0)
        total_bundle_size = overall.get('total_bundle_size', 0)
        avg_api_time = overall.get('avg_api_response_time', 0)
        
        # Performance thresholds
        load_time_status = "✅ FAST" if avg_load_time < 2000 else "⚠️ SLOW" if avg_load_time > 5000 else "✅ GOOD"
        bundle_size_status = "✅ SMALL" if total_bundle_size < 1000000 else "⚠️ LARGE" if total_bundle_size > 5000000 else "✅ MODERATE"
        api_time_status = "✅ FAST" if avg_api_time < 1000 else "⚠️ SLOW" if avg_api_time > 3000 else "✅ GOOD"
        
        summary_table.add_row("Pages Tested", str(overall.get('total_pages_tested', 0)), "📊")
        summary_table.add_row("Avg Load Time", f"{avg_load_time:.0f}ms", load_time_status)
        summary_table.add_row("Max Load Time", f"{max_load_time:.0f}ms", "📊")
        summary_table.add_row("Total Bundle Size", f"{total_bundle_size/1024:.1f}KB", bundle_size_status)
        summary_table.add_row("Avg API Response", f"{avg_api_time:.0f}ms", api_time_status)
        
        console.print(summary_table)
        
        # Per-page breakdown
        if pages_tested:
            console.print(f"\n[bold cyan]📄 Per-Page Performance[/bold cyan]")
            
            page_table = Table(title="Page Performance Details")
            page_table.add_column("Page", style="cyan")
            page_table.add_column("Load Time", style="white")
            page_table.add_column("FCP", style="white")
            page_table.add_column("LCP", style="white")
            page_table.add_column("CLS", style="white")
            page_table.add_column("Mobile Load", style="white")
            
            for page_result in pages_tested:
                desktop = page_result['desktop_metrics']
                mobile = page_result['mobile_metrics']
                
                page_table.add_row(
                    page_result['name'],
                    f"{desktop['page_load_time']:.0f}ms",
                    f"{desktop['first_contentful_paint']:.0f}ms",
                    f"{desktop['largest_contentful_paint']:.0f}ms",
                    f"{desktop['cumulative_layout_shift']:.3f}",
                    f"{mobile['page_load_time']:.0f}ms"
                )
            
            console.print(page_table)
        
        # Performance recommendations
        console.print(f"\n[bold]💡 Performance Recommendations[/bold]")
        
        recommendations = []
        
        if avg_load_time > 3000:
            recommendations.append("Optimize page load times - consider code splitting and lazy loading")
        
        if total_bundle_size > 2000000:  # 2MB
            recommendations.append("Reduce bundle size - remove unused dependencies and optimize assets")
        
        if avg_api_time > 2000:
            recommendations.append("Optimize API response times - add caching and query optimization")
        
        # Bundle-specific recommendations
        for page_result in pages_tested:
            bundle_sizes = page_result['desktop_metrics']['bundle_sizes']
            for bundle, size in bundle_sizes.items():
                if size > 1000000:  # 1MB per bundle
                    recommendations.append(f"Large bundle detected: {bundle} ({size/1024:.1f}KB)")
        
        if recommendations:
            for rec in recommendations:
                console.print(f"[yellow]  • {rec}[/yellow]")
        else:
            console.print("[green]  • Frontend performance looks good![/green]")
        
        # Core Web Vitals assessment
        console.print(f"\n[bold]🎯 Core Web Vitals Assessment[/bold]")
        
        if pages_tested:
            # Average Core Web Vitals across all pages
            all_fcp = [p['desktop_metrics']['first_contentful_paint'] for p in pages_tested]
            all_lcp = [p['desktop_metrics']['largest_contentful_paint'] for p in pages_tested]
            all_cls = [p['desktop_metrics']['cumulative_layout_shift'] for p in pages_tested]
            
            avg_fcp = statistics.mean([x for x in all_fcp if x > 0]) if any(x > 0 for x in all_fcp) else 0
            avg_lcp = statistics.mean([x for x in all_lcp if x > 0]) if any(x > 0 for x in all_lcp) else 0
            avg_cls = statistics.mean([x for x in all_cls if x > 0]) if any(x > 0 for x in all_cls) else 0
            
            # Core Web Vitals status
            fcp_status = "✅ GOOD" if avg_fcp < 1800 else "⚠️ NEEDS IMPROVEMENT" if avg_fcp < 3000 else "❌ POOR"
            lcp_status = "✅ GOOD" if avg_lcp < 2500 else "⚠️ NEEDS IMPROVEMENT" if avg_lcp < 4000 else "❌ POOR"
            cls_status = "✅ GOOD" if avg_cls < 0.1 else "⚠️ NEEDS IMPROVEMENT" if avg_cls < 0.25 else "❌ POOR"
            
            vitals_table = Table(title="Core Web Vitals")
            vitals_table.add_column("Metric", style="cyan")
            vitals_table.add_column("Value", style="white")
            vitals_table.add_column("Status", style="green")
            vitals_table.add_column("Target", style="yellow")
            
            vitals_table.add_row("First Contentful Paint", f"{avg_fcp:.0f}ms", fcp_status, "< 1800ms")
            vitals_table.add_row("Largest Contentful Paint", f"{avg_lcp:.0f}ms", lcp_status, "< 2500ms")
            vitals_table.add_row("Cumulative Layout Shift", f"{avg_cls:.3f}", cls_status, "< 0.1")
            
            console.print(vitals_table)

async def main():
    """Main frontend test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Istanbul Frontend Performance Testing')
    parser.add_argument('--environment', choices=['local', 'production'], 
                       default='local', help='Test environment')
    parser.add_argument('--headless', action='store_true', default=True, 
                       help='Run browser in headless mode')
    
    args = parser.parse_args()
    
    # Configuration
    config = get_config(args.environment)
    frontend_url = config['frontend_url']
    
    console.print(f"[bold]🎨 AI Istanbul Frontend Performance Testing[/bold]")
    console.print(f"[cyan]Environment: {args.environment}[/cyan]")
    console.print(f"[cyan]Frontend URL: {frontend_url}[/cyan]")
    
    # Initialize tester
    tester = FrontendPerformanceTester(frontend_url)
    
    try:
        await tester.setup_browser(headless=args.headless)
        
        # Run comprehensive test
        results = await tester.run_comprehensive_frontend_test()
        
        # Generate report
        tester.generate_frontend_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"frontend_performance_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[green]💾 Frontend performance results saved to: {results_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Frontend test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
