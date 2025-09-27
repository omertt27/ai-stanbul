## Clear Browser Cache and Service Worker

### For Chrome/Chromium:
1. Open DevTools (F12)
2. Go to **Application** tab
3. Click **Service Workers** in left sidebar
4. Click **Unregister** for any ai-istanbul service workers
5. Go to **Storage** tab
6. Click **Clear site data**
7. Hard refresh: **Ctrl+Shift+R** (or **Cmd+Shift+R** on Mac)

### For Firefox:
1. Open DevTools (F12) 
2. Go to **Application** tab
3. Click **Service Workers**
4. **Unregister** any service workers
5. Clear cache: **Ctrl+Shift+Delete** (or **Cmd+Shift+Delete** on Mac)
6. Hard refresh: **Ctrl+F5** (or **Cmd+Shift+R** on Mac)

### Alternative: Open in Incognito/Private Mode
- This bypasses all cached service workers
- **Ctrl+Shift+N** (Chrome) or **Ctrl+Shift+P** (Firefox)
- Navigate to http://localhost:3000

### What I Fixed:
✅ Disabled service worker registration (was interfering with MIME types)
✅ Simplified Vite config with explicit cache control headers
✅ Added esbuild JSX loader configuration
✅ Cleared all Vite and npm caches

The server is now correctly serving `Content-Type: text/javascript` for all module files.
