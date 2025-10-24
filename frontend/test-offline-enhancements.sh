#!/bin/bash

# Offline Enhancements - Quick Test Script
# Run this to verify the implementation is working

echo "🧪 Istanbul AI - Offline Enhancements Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ Error: package.json not found${NC}"
    echo "Please run this script from the frontend directory"
    exit 1
fi

echo -e "${YELLOW}📦 Step 1: Checking implementation files...${NC}"

# Check if files exist
files=(
    "src/services/offlineMapTileCache.js"
    "src/services/offlineIntentDetector.js"
    "src/services/offlineDatabase.js"
    "src/services/offlineEnhancementManager.js"
    "src/components/OfflineEnhancementsUI.jsx"
    "src/pages/OfflineSettings.jsx"
    "src/styles/offline-enhancements.css"
    "public/sw-enhanced.js"
)

missing_files=()
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (MISSING)"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Missing files detected!${NC}"
    echo "Please create the following files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo ""
echo -e "${YELLOW}📝 Step 2: Checking integration...${NC}"

# Check if main.jsx has the initialization
if grep -q "offlineEnhancementManager" src/main.jsx; then
    echo -e "${GREEN}✓${NC} main.jsx - Offline manager imported"
else
    echo -e "${RED}✗${NC} main.jsx - Missing offline manager import"
fi

# Check if AppRouter has the route
if grep -q "offline-settings" src/AppRouter.jsx; then
    echo -e "${GREEN}✓${NC} AppRouter.jsx - Route added"
else
    echo -e "${RED}✗${NC} AppRouter.jsx - Missing route"
fi

# Check if OfflineSettings imports the UI component
if grep -q "OfflineEnhancementsUI" src/pages/OfflineSettings.jsx; then
    echo -e "${GREEN}✓${NC} OfflineSettings.jsx - UI component imported"
else
    echo -e "${RED}✗${NC} OfflineSettings.jsx - Missing UI component"
fi

echo ""
echo -e "${YELLOW}🔧 Step 3: Installing dependencies...${NC}"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm packages..."
    npm install
else
    echo -e "${GREEN}✓${NC} Dependencies already installed"
fi

echo ""
echo -e "${YELLOW}🏗️  Step 4: Building project...${NC}"

# Try to build
npm run build 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Build successful!"
else
    echo -e "${RED}✗${NC} Build failed. Check build.log for details"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "✅ All checks passed!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Start the dev server:"
echo -e "   ${YELLOW}npm run dev${NC}"
echo ""
echo "2. Open browser and navigate to:"
echo -e "   ${YELLOW}http://localhost:5173/offline-settings${NC}"
echo ""
echo "3. Open Chrome DevTools and check:"
echo "   • Application → Service Workers"
echo "   • Application → Cache Storage"
echo "   • Application → IndexedDB"
echo ""
echo "4. Test offline mode:"
echo "   • DevTools → Network → Select 'Offline'"
echo "   • Refresh page"
echo "   • Verify offline features work"
echo ""
echo -e "${YELLOW}📖 For detailed testing steps, see:${NC}"
echo "   OFFLINE_ENHANCEMENTS_NEXT_STEPS.md"
echo ""
