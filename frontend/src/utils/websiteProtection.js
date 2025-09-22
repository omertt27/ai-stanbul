// Website Protection and Anti-Copy Measures
// This file implements various techniques to protect the AI Istanbul website

class WebsiteProtection {
    constructor() {
        this.init();
    }

    init() {
        this.disableRightClick();
        this.disableKeyboardShortcuts();
        this.disableTextSelection();
        this.disableImageDragging();
        this.addCopyrightNotices();
        // Developer tools detection disabled for better user experience
        // this.detectDevTools();
        this.obfuscateSourceCode();
        this.addWatermarks();
        this.preventScreenshots();
        this.blockCommonScrapingUserAgents();
        this.addAntiDebugger();
        this.protectConsole();
    }

    // Disable right-click context menu (silent protection)
    disableRightClick() {
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            // Silent protection - no warning popup
            return false;
        });
    }

    // Disable common keyboard shortcuts for copying/inspecting
    disableKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Disable F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U, Ctrl+A, Ctrl+S, Ctrl+C
            if (
                e.key === 'F12' ||
                (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'J')) ||
                (e.ctrlKey && e.key === 'u') ||
                (e.ctrlKey && e.key === 'a') ||
                (e.ctrlKey && e.key === 's') ||
                (e.ctrlKey && e.key === 'c') ||
                (e.ctrlKey && e.shiftKey && e.key === 'C')
            ) {
                e.preventDefault();
                // Silent protection - no warning popup
                return false;
            }
        });
    }

    // Disable text selection
    disableTextSelection() {
        document.onselectstart = () => false;
        document.onmousedown = () => false;
        
        // Add CSS to prevent selection
        const style = document.createElement('style');
        style.textContent = `
            * {
                -webkit-user-select: none !important;
                -moz-user-select: none !important;
                -ms-user-select: none !important;
                user-select: none !important;
                -webkit-touch-callout: none !important;
                -webkit-tap-highlight-color: transparent !important;
            }
            input, textarea {
                -webkit-user-select: text !important;
                -moz-user-select: text !important;
                -ms-user-select: text !important;
                user-select: text !important;
            }
        `;
        document.head.appendChild(style);
    }

    // Disable image dragging
    disableImageDragging() {
        document.addEventListener('dragstart', (e) => {
            if (e.target.tagName === 'IMG') {
                e.preventDefault();
                return false;
            }
        });
    }

    // Add copyright notices
    addCopyrightNotices() {
        // Add copyright to console
        console.clear();
        console.log('%c🔒 AI Istanbul - Copyright Protected', 'color: red; font-size: 20px; font-weight: bold;');
        console.log('%c⚠️ Unauthorized copying or reproduction is prohibited', 'color: orange; font-size: 14px;');
        console.log('%c📞 Contact: legal@ai-istanbul.com for licensing', 'color: blue; font-size: 12px;');
        
        // Add invisible watermarks to DOM
        const watermark = document.createElement('div');
        watermark.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><text x="50%" y="50%" text-anchor="middle" fill="rgba(255,0,0,0.1)" font-size="12">© AI Istanbul ${new Date().getFullYear()}</text></svg>') repeat;
            opacity: 0.05;
        `;
        document.body.appendChild(watermark);
    }

    // Developer tools detection disabled to avoid user warnings
    detectDevTools() {
        // Detection disabled for better user experience
        return false;
    }

    // Show warning for developer tools
    showDevToolsWarning() {
        if (document.getElementById('devtools-warning')) return;
        
        const warning = document.createElement('div');
        warning.id = 'devtools-warning';
        warning.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white;
            padding: 30px;
            border-radius: 15px;
            z-index: 999999;
            font-family: Arial, sans-serif;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 3px solid #fff;
        `;
        warning.innerHTML = `
            <h2>🚫 DEVELOPER TOOLS DETECTED</h2>
            <p>⚠️ This website is protected by copyright</p>
            <p>🔒 Unauthorized copying is prohibited</p>
            <p>📧 Contact: legal@ai-istanbul.com</p>
            <button onclick="this.parentElement.remove()" style="
                background: white;
                color: #cc0000;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                margin-top: 10px;
            ">I Understand</button>
        `;
        document.body.appendChild(warning);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (warning.parentElement) {
                warning.remove();
            }
        }, 10000);
    }

    // Show general warning (disabled for better user experience)
    showWarning(message) {
        // Warnings disabled to avoid interrupting user experience
        console.log('Protection notice:', message);
        return;
    }

    // Basic code obfuscation
    obfuscateSourceCode() {
        // Add fake comments to confuse scrapers
        const fakeComments = [
            '<!-- Fake API endpoint: https://fake-api.example.com -->',
            '<!-- Decoy database: mongodb://fake-db:27017 -->',
            '<!-- Dummy secret key: sk-fake-key-12345 -->',
            '<!-- Red herring: This is not the real implementation -->'
        ];
        
        fakeComments.forEach(comment => {
            document.head.insertAdjacentHTML('beforeend', comment);
        });
    }

    // Prevent screenshots and screen recording
    preventScreenshots() {
        // Add CSS to prevent print screen
        const style = document.createElement('style');
        style.textContent = `
            @media print {
                * { 
                    visibility: hidden !important; 
                }
                body::before {
                    content: "© AI Istanbul - This content is protected by copyright and cannot be printed or copied.";
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 24px;
                    color: red;
                    visibility: visible !important;
                }
            }
        `;
        document.head.appendChild(style);

        // Detect print attempts (silent protection)
        window.addEventListener('beforeprint', (e) => {
            e.preventDefault();
            // Silent protection - no warning popup
            return false;
        });

        // Try to detect screen recording software
        let lastTime = performance.now();
        setInterval(() => {
            const currentTime = performance.now();
            if (currentTime - lastTime > 100) {
                // Possible screen recording or debugging
                // Silent protection - no warning popup
                console.log('Suspicious activity detected');
            }
            lastTime = currentTime;
        }, 50);
    }

    // Block common scraping user agents
    blockCommonScrapingUserAgents() {
        const userAgent = navigator.userAgent.toLowerCase();
        const blockedAgents = [
            'bot', 'crawl', 'spider', 'scrape', 'selenium', 'puppeteer', 
            'playwright', 'phantom', 'wget', 'curl', 'requests', 'scrapy'
        ];
        
        for (const agent of blockedAgents) {
            if (userAgent.includes(agent)) {
                document.body.innerHTML = `
                    <div style="padding: 50px; text-align: center; color: red; font-family: Arial;">
                        <h1>🚫 Access Denied</h1>
                        <p>Automated access is not permitted</p>
                        <p>© AI Istanbul - Protected Content</p>
                    </div>
                `;
                throw new Error('Automated access blocked');
            }
        }
    }

    // Add anti-debugger techniques
    addAntiDebugger() {
        // Anti-debugging with infinite debugger loop
        setInterval(() => {
            if (window.console && (window.console.firebug || window.console.exception)) {
                try {
                    debugger;
                } catch (e) {}
            }
        }, 1000);

        // Detect if running in a debugger
        let start = performance.now();
        debugger;
        let end = performance.now();
        if (end - start > 100) {
            this.showDevToolsWarning();
        }
    }

    // Protect console methods
    protectConsole() {
        const originalMethods = {};
        
        // Store original methods
        ['log', 'warn', 'error', 'info', 'debug'].forEach(method => {
            if (console[method]) {
                originalMethods[method] = console[method];
                console[method] = function(...args) {
                    // Add copyright notice to all console outputs
                    originalMethods[method].apply(console, ['🔒 AI Istanbul Protected:', ...args]);
                };
            }
        });

        // Override toString methods to hide implementation
        Object.defineProperty(Function.prototype, 'toString', {
            value: function() {
                return 'function() { [Protected Code] }';
            },
            writable: false,
            configurable: false
        });
    }

    // Add watermarks to content
    addWatermarks() {
        // Add invisible text watermarks
        const textNodes = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let node;
        let count = 0;
        while (node = textNodes.nextNode()) {
            if (count % 50 === 0 && node.textContent.trim().length > 20) {
                node.textContent += '​'; // Zero-width space watermark
            }
            count++;
        }
    }
}

// Initialize protection when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new WebsiteProtection();
    });
} else {
    new WebsiteProtection();
}

// Additional protection for React/Vite apps
window.addEventListener('load', () => {
    // Re-initialize after React renders
    setTimeout(() => {
        new WebsiteProtection();
    }, 1000);
});

export default WebsiteProtection;
