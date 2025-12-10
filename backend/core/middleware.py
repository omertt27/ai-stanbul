"""
Middleware Configuration Module

CORS, logging, error handling, and other middleware
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI):
    """Configure CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("✅ CORS middleware configured")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"→ {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"← {request.method} {request.url.path} "
                f"[{response.status_code}] {duration:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"✗ {request.method} {request.url.path} "
                f"failed after {duration:.3f}s: {str(e)}"
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "message": str(e) if settings.is_production() else str(e),
                    "type": type(e).__name__
                }
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers including CSP"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Content Security Policy - Complete configuration for maps, CDNs, and analytics
        csp_directives = [
            "default-src 'self'",
            "frame-src 'self' https://vercel.live https://*.vercel.live https://vercel.com",
            
            # Connect-src: APIs, WebSockets, Map tiles, Analytics
            "connect-src 'self' "
            "https://ai-stanbul.onrender.com https://aistanbul.net "
            "https://images.unsplash.com https://*.unsplash.com "
            # OpenStreetMap tiles
            "https://*.tile.openstreetmap.org https://tile.openstreetmap.org "
            "https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org "
            # CARTO/CartoDB map tiles
            "https://*.basemaps.cartocdn.com https://basemaps.cartocdn.com "
            # CDN sources
            "https://cdnjs.cloudflare.com https://*.cdnjs.cloudflare.com "
            "https://unpkg.com https://*.unpkg.com "
            # Analytics
            "https://www.google-analytics.com https://ssl.google-analytics.com "
            "https://www.googletagmanager.com https://analytics.google.com "
            "https://region1.google-analytics.com https://region1.analytics.google.com "
            "https://*.google-analytics.com https://*.analytics.google.com "
            "https://cdn.amplitude.com "
            # Other services
            "https://maps.googleapis.com https://fonts.googleapis.com https://fonts.gstatic.com "
            "https://vercel.live https://*.vercel.live https://vercel.com https://*.vercel.app "
            "wss://vercel.live wss://*.vercel.live",
            
            # Img-src: Images, map tiles, and marker icons
            "img-src 'self' "
            "https://images.unsplash.com https://*.unsplash.com "
            # OpenStreetMap tiles
            "https://*.tile.openstreetmap.org https://tile.openstreetmap.org "
            "https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org "
            # CARTO/CartoDB tiles
            "https://*.basemaps.cartocdn.com https://basemaps.cartocdn.com "
            # CDN for Leaflet marker icons
            "https://cdnjs.cloudflare.com https://*.cdnjs.cloudflare.com "
            "https://unpkg.com https://*.unpkg.com "
            # GitHub raw content for custom markers
            "https://raw.githubusercontent.com "
            "data: blob:",
            
            # Script-src: Allow analytics, tracking, and CDN scripts
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://vercel.live https://*.vercel.live "
            "https://www.googletagmanager.com https://www.google-analytics.com "
            "https://cdn.amplitude.com "
            "https://cdnjs.cloudflare.com https://unpkg.com "
            "https://*.vercel.app",
            
            # Style-src: Allow inline styles and CDN stylesheets
            "style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com "
            "https://cdnjs.cloudflare.com https://unpkg.com",
            
            "font-src 'self' https://fonts.gstatic.com data:",
            "media-src 'self' blob:",
            "worker-src 'self' blob:",
            "manifest-src 'self'",
            "base-uri 'self'"
        ]
        
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


def setup_middleware(app: FastAPI):
    """Setup all middleware"""
    setup_cors(app)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    logger.info("✅ Middleware configured")
