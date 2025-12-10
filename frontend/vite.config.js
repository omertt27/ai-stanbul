import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: false,
    headers: {
      'Permissions-Policy': 'geolocation=(self)',
      'Content-Security-Policy': [
        "default-src 'self'",
        "connect-src 'self' http://localhost:5001 http://localhost:8001 https://ai-stanbul.onrender.com https://*.tile.openstreetmap.org https://tile.openstreetmap.org https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://basemaps.cartocdn.com https://cdnjs.cloudflare.com https://unpkg.com https://www.google-analytics.com https://www.googletagmanager.com https://cdn.amplitude.com https://region1.analytics.google.com",
        "img-src 'self' data: blob: https://*.tile.openstreetmap.org https://tile.openstreetmap.org https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://basemaps.cartocdn.com https://cdnjs.cloudflare.com https://unpkg.com https://images.unsplash.com https://*.unsplash.com",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.googletagmanager.com https://www.google-analytics.com https://cdn.amplitude.com https://cdnjs.cloudflare.com https://unpkg.com",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://unpkg.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        "worker-src 'self' blob:",
        "frame-src 'self'"
      ].join('; ')
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: undefined,
        // Ensure consistent asset naming with cache busting
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  },
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.[jt]sx?$/
  },
  // Fix for SPA routing - ensure index.html is served for all routes
  preview: {
    port: 3000,
    strictPort: false,
    host: '0.0.0.0'
  }
})
