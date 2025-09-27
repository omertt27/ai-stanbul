import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  server: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: false,
    hmr: {
      overlay: false
    },
    headers: {
      'Cache-Control': 'no-cache, no-store, must-revalidate'
    },
    middlewareMode: false,
    fs: {
      strict: false
    }
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify('development')
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom']
  }
})
