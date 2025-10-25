#!/bin/bash
# 🚀 Enhanced Authentication - Quick Integration Script
# This script helps integrate authentication into backend/main.py in 15 minutes

set -e  # Exit on error

echo "🔒 Enhanced Authentication Integration Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo -e "${RED}❌ Error: backend/main.py not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo -e "${BLUE}📍 Current directory: $(pwd)${NC}"
echo ""

# Step 1: Install dependencies
echo -e "${GREEN}Step 1: Installing dependencies...${NC}"
echo -e "${YELLOW}Running: pip install PyJWT passlib[bcrypt] email-validator${NC}"

if pip install PyJWT==2.8.0 passlib[bcrypt]==1.7.4 email-validator==2.1.0 --quiet; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Dependency installation had warnings (may already be installed)${NC}"
fi
echo ""

# Step 2: Check if enhanced_auth.py exists
echo -e "${GREEN}Step 2: Checking authentication module...${NC}"
if [ ! -f "backend/enhanced_auth.py" ]; then
    echo -e "${RED}❌ Error: backend/enhanced_auth.py not found${NC}"
    echo "Please ensure the enhanced authentication module exists"
    exit 1
fi
echo -e "${GREEN}✅ Authentication module found${NC}"
echo ""

# Step 3: Create backup of main.py
echo -e "${GREEN}Step 3: Creating backup...${NC}"
BACKUP_FILE="backend/main.py.backup_$(date +%Y%m%d_%H%M%S)"
cp backend/main.py "$BACKUP_FILE"
echo -e "${GREEN}✅ Backup created: $BACKUP_FILE${NC}"
echo ""

# Step 4: Check if authentication is already integrated
echo -e "${GREEN}Step 4: Checking integration status...${NC}"
if grep -q "from enhanced_auth import" backend/main.py 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Authentication appears to already be integrated${NC}"
    echo -e "${YELLOW}   Found 'from enhanced_auth import' in main.py${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}ℹ️  Integration cancelled${NC}"
        exit 0
    fi
fi
echo ""

# Step 5: Create integration snippet file
echo -e "${GREEN}Step 5: Creating integration code...${NC}"
cat > /tmp/auth_integration_snippet.py << 'EOF'
# ============================================
# Enhanced Authentication Integration
# Added by quick integration script
# ============================================

from enhanced_auth import (
    EnhancedAuthManager,
    get_current_user,
    UserRegistrationRequest,
    UserLoginRequest,
    UserRefreshRequest,
    TokenResponse,
    UserResponse
)

# Initialize authentication manager (add this near other initializations)
auth_manager = None
if redis_client:
    try:
        auth_manager = EnhancedAuthManager(redis_client=redis_client)
        print("✅ Enhanced Authentication System initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize authentication: {e}")
        auth_manager = None
else:
    print("⚠️ Authentication requires Redis - disabled")

# ============================================
# Authentication Endpoints
# Add these endpoints to your FastAPI app
# ============================================

@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register(request: UserRegistrationRequest):
    """Register a new user"""
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    
    result = auth_manager.register_user(
        username=request.username,
        password=request.password,
        email=request.email
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result["user"]


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: UserLoginRequest):
    """Login and get JWT tokens"""
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    
    result = auth_manager.login_user(
        username=request.username,
        password=request.password,
        remember_me=request.remember_me
    )
    
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    
    return TokenResponse(**result["tokens"])


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(request: UserRefreshRequest):
    """Refresh access token using refresh token"""
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    
    result = auth_manager.refresh_access_token(request.refresh_token)
    
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    
    return TokenResponse(**result["tokens"])


@app.post("/auth/logout", tags=["Authentication"])
async def logout(current_user: Dict = Depends(get_current_user)):
    """Logout and invalidate tokens"""
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    
    # Get token from request (it's in the dependency)
    # You can extract it if needed, or just return success
    return {"message": "Logged out successfully"}


@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        user_id=current_user["sub"],
        username=current_user["username"],
        email=current_user.get("email")
    )

# ============================================
# Optional: Protected Chat Endpoint Example
# ============================================
# To protect the chat endpoint, add this parameter:
# async def chat(..., current_user: Dict = Depends(get_current_user)):
#     user_id = current_user["sub"]
#     # Bind chat session to user
#     if auth_manager:
#         auth_manager.bind_chat_session_to_user(session_id, user_id)
# ============================================

EOF

echo -e "${GREEN}✅ Integration snippet created${NC}"
echo ""

# Step 6: Show manual integration instructions
echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}📝 MANUAL INTEGRATION REQUIRED${NC}"
echo -e "${YELLOW}============================================${NC}"
echo ""
echo -e "${BLUE}The integration code has been prepared in:${NC}"
echo -e "${GREEN}/tmp/auth_integration_snippet.py${NC}"
echo ""
echo -e "${BLUE}To complete the integration:${NC}"
echo ""
echo "1. Open backend/main.py in your editor"
echo ""
echo "2. Find where Redis client is initialized (around line 700-800)"
echo "   Look for: redis_client = redis.Redis(...)"
echo ""
echo "3. Add the authentication imports at the top of main.py:"
echo -e "${GREEN}   from enhanced_auth import (${NC}"
echo -e "${GREEN}       EnhancedAuthManager,${NC}"
echo -e "${GREEN}       get_current_user,${NC}"
echo -e "${GREEN}       UserRegistrationRequest,${NC}"
echo -e "${GREEN}       UserLoginRequest,${NC}"
echo -e "${GREEN}       TokenResponse${NC}"
echo -e "${GREEN}   )${NC}"
echo ""
echo "4. Initialize auth_manager after redis_client initialization:"
echo -e "${GREEN}   auth_manager = EnhancedAuthManager(redis_client=redis_client)${NC}"
echo ""
echo "5. Add the authentication endpoints (copy from /tmp/auth_integration_snippet.py)"
echo "   Add them after the health check endpoints (around line 1900+)"
echo ""
echo "6. (Optional) Protect chat endpoint by adding:"
echo -e "${GREEN}   async def chat(..., current_user: Dict = Depends(get_current_user)):${NC}"
echo ""
echo -e "${YELLOW}============================================${NC}"
echo ""

# Step 7: Offer to open files
echo -e "${BLUE}Would you like to:${NC}"
echo "1. View the integration snippet"
echo "2. Open main.py for editing"
echo "3. See detailed integration guide"
echo "4. Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}Integration snippet:${NC}"
        cat /tmp/auth_integration_snippet.py
        ;;
    2)
        echo ""
        echo -e "${GREEN}Opening main.py...${NC}"
        ${EDITOR:-nano} backend/main.py
        ;;
    3)
        echo ""
        echo -e "${GREEN}Opening integration guide...${NC}"
        if [ -f "AUTHENTICATION_INTEGRATION_GUIDE.md" ]; then
            ${EDITOR:-less} AUTHENTICATION_INTEGRATION_GUIDE.md
        else
            echo -e "${YELLOW}⚠️  Integration guide not found${NC}"
        fi
        ;;
    4)
        echo -e "${BLUE}ℹ️  Exiting${NC}"
        ;;
    *)
        echo -e "${YELLOW}⚠️  Invalid choice${NC}"
        ;;
esac

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Quick Integration Setup Complete${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Follow the manual integration instructions above"
echo "2. Test authentication endpoints:"
echo "   curl -X POST http://localhost:8000/auth/register -H 'Content-Type: application/json' -d '{\"username\":\"test\",\"password\":\"Test123!\",\"email\":\"test@example.com\"}'"
echo "3. Check AUTHENTICATION_INTEGRATION_GUIDE.md for details"
echo ""
echo -e "${YELLOW}📄 Backup created: $BACKUP_FILE${NC}"
echo -e "${YELLOW}📄 Integration snippet: /tmp/auth_integration_snippet.py${NC}"
echo ""
echo -e "${GREEN}🎉 Ready to integrate!${NC}"
