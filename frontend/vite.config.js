import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: false,
    // Temporarily disable CSP for local development to avoid connection issues
    // Re-enable for production builds
    // headers: {
    //   'Permissions-Policy': 'geolocation=(self)',
    //   'Content-Security-Policy': [
    //     "default-src 'self'",
    //     "connect-src 'self' http://localhost:8000 http://localhost:5001 http://localhost:8001 https://ai-stanbul.onrender.com https://ai-stanbul-509659445005.europe-west1.run.app https://*.tile.openstreetmap.org https://tile.openstreetmap.org https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://basemaps.cartocdn.com https://cdnjs.cloudflare.com https://unpkg.com https://www.google-analytics.com https://www.googletagmanager.com https://cdn.amplitude.com https://api2.amplitude.com https://region1.analytics.google.com",
    //     "img-src 'self' data: blob: https://*.tile.openstreetmap.org https://tile.openstreetmap.org https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://basemaps.cartocdn.com https://cdnjs.cloudflare.com https://unpkg.com https://images.unsplash.com https://*.unsplash.com",
    //     "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.googletagmanager.com https://www.google-analytics.com https://cdn.amplitude.com https://cdnjs.cloudflare.com https://unpkg.com",
    //     "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://unpkg.com",
    //     "font-src 'self' https://fonts.gstatic.com data:",
    //     "worker-src 'self' blob:",
    //     "frame-src 'self'"
    //   ].join('; ')
    // }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    // Force new build hash - CRITICAL for cache busting
    rollupOptions: {
      output: {
        // Smart code splitting for better caching
        manualChunks(id) {
          // Vendor chunks
          if (id.includes('node_modules')) {
            // Separate large libraries
            if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) {
              return 'vendor-react';
            }
            if (id.includes('leaflet') || id.includes('react-leaflet')) {
              return 'vendor-maps';
            }
            if (id.includes('i18next') || id.includes('react-i18next')) {
              return 'vendor-i18n';
            }
            if (id.includes('@sentry')) {
              return 'vendor-sentry';
            }
            // All other vendor code
            return 'vendor';
          }
          
          // App chunks
          if (id.includes('/pages/')) {
            return 'pages';
          }
          if (id.includes('/components/')) {
            return 'components';
          }
          if (id.includes('/services/')) {
            return 'services';
          }
        },
        // Ensure consistent asset naming with cache busting
        // Using timestamp in dev to force new hashes
        entryFileNames: `assets/[name]-[hash]-${Date.now()}.js`,
        chunkFileNames: `assets/[name]-[hash]-${Date.now()}.js`,
        assetFileNames: `assets/[name]-[hash]-${Date.now()}.[ext]`
      }
    },
    // Optimize chunk size
    chunkSizeWarningLimit: 1000, // Increase limit for map libraries
    minify: 'terser', // Use terser instead of esbuild for better variable hoisting
    terserOptions: {
      compress: {
        drop_console: false,
        pure_funcs: [],
        passes: 1, // Single pass to avoid aggressive optimization
        sequences: false, // Don't join consecutive statements
        properties: false, // Don't optimize property access
        dead_code: true,
        drop_debugger: true,
        conditionals: true,
        evaluate: false, // Don't evaluate constant expressions
        booleans: false, // Don't optimize boolean expressions
        loops: false, // Don't optimize loops
        unused: true,
        toplevel: false,
        if_return: false,
        inline: false, // CRITICAL: Don't inline function calls
        join_vars: false, // CRITICAL: Don't join variable declarations
        collapse_vars: false, // CRITICAL: Don't collapse single-use variables
        reduce_vars: false, // CRITICAL: Don't reduce variables
        warnings: false,
        negate_iife: false,
        pure_getters: false,
        keep_fargs: true,
        keep_fnames: true,
        keep_classnames: true,
        hoist_funs: false, // CRITICAL: Don't hoist function declarations
        hoist_props: false, // CRITICAL: Don't hoist properties
        hoist_vars: false, // CRITICAL: Don't hoist var declarations
        side_effects: false // CRITICAL: Assume all code has side effects
      },
      mangle: false, // CRITICAL: Disable all variable name mangling
      format: {
        comments: false,
        beautify: false
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
