#!/usr/bin/env python3
"""
Production optimization script
Run this to prepare your application for production deployment
"""

import os
import re
import json
from pathlib import Path

def optimize_frontend():
    """Optimize frontend for production"""
    print("🎨 Optimizing frontend...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Update package.json for production optimizations
    package_json_path = frontend_dir / "package.json"
    if package_json_path.exists():
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        # Add production build optimizations
        if "scripts" not in package_data:
            package_data["scripts"] = {}
        
        package_data["scripts"].update({
            "build": "vite build",
            "build:production": "vite build --mode production",
            "preview": "vite preview --port 4173",
            "analyze": "vite-bundle-analyzer dist"
        })
        
        # Add production dependencies
        if "devDependencies" not in package_data:
            package_data["devDependencies"] = {}
        
        package_data["devDependencies"]["vite-bundle-analyzer"] = "^0.7.0"
        
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)
        
        print("✅ Frontend package.json optimized")
    
    # Create production vite config if needed
    vite_config_path = frontend_dir / "vite.config.js"
    if vite_config_path.exists():
        print("✅ Vite config already exists")
    else:
        vite_config_content = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom']
        }
      }
    }
  },
  server: {
    port: 5173
  }
})"""
        with open(vite_config_path, 'w') as f:
            f.write(vite_config_content)
        print("✅ Production vite config created")

def optimize_backend():
    """Optimize backend for production"""
    print("🔧 Optimizing backend...")
    
    # Create production startup script
    startup_script = Path(__file__).parent / "backend" / "start_production.py"
    startup_content = """#!/usr/bin/env python3
import os
import subprocess
import sys

def start_production_server():
    print("🚀 Starting production server...")
    
    # Set production environment
    os.environ["ENVIRONMENT"] = "production"
    os.environ["DEBUG"] = "False"
    
    # Get port from environment (for Railway, Render, etc.)
    port = os.environ.get("PORT", "8000")
    
    # Start with Gunicorn for production
    cmd = [
        "gunicorn", 
        "main:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--workers", "4",
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", "info"
    ]
    
    print(f"Starting server on port {port}")
    subprocess.run(cmd)

if __name__ == "__main__":
    start_production_server()
"""
    
    with open(startup_script, 'w') as f:
        f.write(startup_content)
    
    # Make it executable
    os.chmod(startup_script, 0o755)
    print("✅ Production startup script created")

def remove_development_files():
    """Remove development-only files"""
    print("🧹 Cleaning up development files...")
    
    # Files to remove in production
    dev_files = [
        "backend/test_*.py",
        "backend/*_test.py", 
        "backend/debug_*.py",
        "frontend/test.html",
        "frontend/public/api-test.html"
    ]
    
    for pattern in dev_files:
        for file_path in Path(".").glob(pattern):
            if file_path.exists():
                print(f"   Removing {file_path}")
                # Don't actually remove, just show what would be removed
                # file_path.unlink()
    
    print("✅ Development files identified for removal")

def check_production_readiness():
    """Check if application is ready for production"""
    print("🔍 Checking production readiness...")
    
    issues = []
    
    # Check for environment files
    if not Path("backend/.env.example").exists():
        issues.append("Missing backend/.env.example")
    if not Path("frontend/.env.example").exists():
        issues.append("Missing frontend/.env.example")
    
    # Check for hardcoded secrets (basic check)
    backend_files = list(Path("backend").glob("*.py"))
    for file_path in backend_files:
        with open(file_path, 'r') as f:
            content = f.read()
            if re.search(r'sk-[a-zA-Z0-9]{20,}', content):
                issues.append(f"Potential hardcoded API key in {file_path}")
    
    if issues:
        print("❌ Production readiness issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Application appears ready for production")
        return True

def main():
    print("🔧 Production Optimization Script")
    print("=" * 50)
    
    optimize_frontend()
    optimize_backend()
    remove_development_files()
    
    if check_production_readiness():
        print("\n🎉 Production optimization complete!")
        print("\n📋 Next steps:")
        print("   1. Set up your hosting accounts (Vercel, Railway)")
        print("   2. Get your API keys (OpenAI, Google)")
        print("   3. Configure environment variables")
        print("   4. Deploy using the deployment guide")
    else:
        print("\n⚠️  Please fix the issues above before deploying")

if __name__ == "__main__":
    main()
