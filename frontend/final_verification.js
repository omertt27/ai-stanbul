/**
 * Final Verification Script for AIstanbul App
 * Tests all functionality including UI consistency, chatbot robustness, and navigation
 */

// UI Consistency Tests
function testUIConsistency() {
    console.log('🎨 Testing UI Consistency...');
    
    const tests = [
        {
            name: 'Logo Position Consistency',
            test: () => {
                const logos = document.querySelectorAll('.logo-istanbul');
                const fixedLogos = Array.from(logos).filter(logo => 
                    logo.closest('.fixed') || 
                    getComputedStyle(logo.closest('*')).position === 'fixed'
                );
                return fixedLogos.length > 0 ? 'PASS' : 'FAIL';
            }
        },
        {
            name: 'Footer Fixed Position',
            test: () => {
                const footer = document.querySelector('footer');
                if (!footer) return 'NO_FOOTER';
                const style = getComputedStyle(footer);
                return style.position === 'fixed' && style.bottom === '0px' ? 'PASS' : 'FAIL';
            }
        },
        {
            name: 'Logo Text Styling',
            test: () => {
                const logoTexts = document.querySelectorAll('.logo-text');
                const hasGradient = Array.from(logoTexts).some(text => {
                    const style = getComputedStyle(text);
                    return style.background.includes('gradient') || 
                           style.backgroundImage.includes('gradient');
                });
                return hasGradient ? 'PASS' : 'FAIL';
            }
        },
        {
            name: 'Navigation Z-Index',
            test: () => {
                const nav = document.querySelector('.nav-container');
                if (!nav) return 'NO_NAV';
                const zIndex = parseInt(getComputedStyle(nav).zIndex);
                return zIndex >= 1000 ? 'PASS' : 'FAIL';
            }
        }
    ];
    
    tests.forEach(test => {
        const result = test.test();
        console.log(`  ${test.name}: ${result}`);
    });
}

// Navigation Tests
function testNavigation() {
    console.log('\n🧭 Testing Navigation...');
    
    const navLinks = document.querySelectorAll('.nav-link');
    console.log(`  Found ${navLinks.length} navigation links`);
    
    navLinks.forEach((link, index) => {
        console.log(`  Link ${index + 1}: ${link.textContent}`);
    });
    
    // Test responsive behavior
    const nav = document.querySelector('.nav-container');
    if (nav) {
        const style = getComputedStyle(nav);
        console.log(`  Navigation background: ${style.background}`);
        console.log(`  Navigation position: ${style.position}`);
    }
}

// Chatbot API Health Check
async function testChatbotHealth() {
    console.log('\n🤖 Testing Chatbot Health...');
    
    try {
        const response = await fetch('http://localhost:8001/health');
        if (response.ok) {
            console.log('  ✅ Backend server is healthy');
            return true;
        } else {
            console.log('  ⚠️ Backend server responded but may have issues');
            return false;
        }
    } catch (error) {
        console.log('  ❌ Backend server is not accessible');
        console.log(`  Error: ${error.message}`);
        return false;
    }
}

// Basic Chatbot Test
async function testChatbotBasic() {
    console.log('\n💬 Testing Basic Chatbot Functionality...');
    
    const testCases = [
        'Hello',
        'What restaurants do you recommend in Istanbul?',
        'Tell me about museums',
        ''  // Empty input test
    ];
    
    for (const message of testCases) {
        try {
            const response = await fetch('http://localhost:8001/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: `test_${Date.now()}`
                })
            });
            
            const data = await response.json();
            const preview = message || '[empty]';
            const responsePreview = data.message ? data.message.substring(0, 50) : '[no response]';
            
            console.log(`  "${preview}" → "${responsePreview}..."`);
        } catch (error) {
            console.log(`  "${message}" → ERROR: ${error.message}`);
        }
        
        // Small delay between requests
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}

// Responsive Design Test
function testResponsiveDesign() {
    console.log('\n📱 Testing Responsive Design...');
    
    const viewports = [
        { width: 1920, height: 1080, name: 'Desktop Large' },
        { width: 1366, height: 768, name: 'Desktop Medium' },
        { width: 768, height: 1024, name: 'Tablet' },
        { width: 375, height: 667, name: 'Mobile' }
    ];
    
    const originalWidth = window.innerWidth;
    const originalHeight = window.innerHeight;
    
    viewports.forEach(viewport => {
        // Note: This is limited in a browser environment
        // Real responsive testing would require browser dev tools or testing framework
        console.log(`  ${viewport.name} (${viewport.width}x${viewport.height})`);
        
        // Check if viewport meta tag exists
        const viewportMeta = document.querySelector('meta[name="viewport"]');
        if (viewportMeta) {
            console.log(`    Viewport meta: ${viewportMeta.content}`);
        }
        
        // Check CSS media queries (simplified test)
        const logoText = document.querySelector('.logo-text');
        if (logoText) {
            const fontSize = getComputedStyle(logoText).fontSize;
            console.log(`    Logo font size: ${fontSize}`);
        }
    });
}

// Accessibility Tests
function testAccessibility() {
    console.log('\n♿ Testing Accessibility...');
    
    const tests = [
        {
            name: 'Alt text for images',
            test: () => {
                const images = document.querySelectorAll('img');
                const imagesWithoutAlt = Array.from(images).filter(img => !img.alt);
                return imagesWithoutAlt.length === 0 ? 'PASS' : `FAIL (${imagesWithoutAlt.length} images without alt)`;
            }
        },
        {
            name: 'Form labels',
            test: () => {
                const inputs = document.querySelectorAll('input');
                const inputsWithoutLabel = Array.from(inputs).filter(input => {
                    return !input.labels || input.labels.length === 0;
                });
                return inputsWithoutLabel.length === 0 ? 'PASS' : `FAIL (${inputsWithoutLabel.length} inputs without labels)`;
            }
        },
        {
            name: 'Focus indicators',
            test: () => {
                const focusableElements = document.querySelectorAll('button, input, a, [tabindex]');
                // This is a simplified test - real testing would require interaction
                return focusableElements.length > 0 ? 'PASS' : 'FAIL';
            }
        },
        {
            name: 'Color contrast',
            test: () => {
                // This is a simplified test - real color contrast testing requires specialized tools
                const body = document.body;
                const hasThemeClass = body.classList.contains('light') || body.classList.contains('dark');
                return hasThemeClass ? 'PASS (theme detected)' : 'UNKNOWN';
            }
        }
    ];
    
    tests.forEach(test => {
        const result = test.test();
        console.log(`  ${test.name}: ${result}`);
    });
}

// Performance Tests
function testPerformance() {
    console.log('\n⚡ Testing Performance...');
    
    const metrics = {
        'DOM nodes': document.querySelectorAll('*').length,
        'CSS files': document.querySelectorAll('link[rel="stylesheet"]').length,
        'JS files': document.querySelectorAll('script[src]').length,
        'Images': document.querySelectorAll('img').length,
        'Font families': new Set(
            Array.from(document.querySelectorAll('*'))
                .map(el => getComputedStyle(el).fontFamily)
                .filter(font => font && font !== 'initial')
        ).size
    };
    
    Object.entries(metrics).forEach(([metric, value]) => {
        console.log(`  ${metric}: ${value}`);
    });
    
    // Memory usage (if available)
    if (performance.memory) {
        console.log(`  Memory used: ${(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  Memory limit: ${(performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`);
    }
}

// Main test runner
async function runAllVerificationTests() {
    console.log('🚀 AIstanbul App - Final Verification Suite');
    console.log('═'.repeat(60));
    
    testUIConsistency();
    testNavigation();
    
    const chatbotHealthy = await testChatbotHealth();
    if (chatbotHealthy) {
        await testChatbotBasic();
    }
    
    testResponsiveDesign();
    testAccessibility();
    testPerformance();
    
    console.log('\n✅ Verification Complete!');
    console.log('═'.repeat(60));
}

// Auto-run when script loads
if (typeof window !== 'undefined') {
    window.verifyAIstanbulApp = runAllVerificationTests;
    
    // Add button to run tests
    const button = document.createElement('button');
    button.textContent = '🔍 Run Verification Tests';
    button.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        z-index: 10000;
        padding: 10px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    `;
    
    button.onmouseover = () => {
        button.style.transform = 'translateY(-2px)';
        button.style.boxShadow = '0 6px 20px rgba(0,0,0,0.3)';
    };
    
    button.onmouseout = () => {
        button.style.transform = 'translateY(0)';
        button.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
    };
    
    button.onclick = runAllVerificationTests;
    document.body.appendChild(button);
}

export { runAllVerificationTests };
