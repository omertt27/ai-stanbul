#!/bin/bash
# Quick Git History Cleaner for Exposed Secrets
# This script removes sensitive data from git history

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🧹 AI ISTANBUL - GIT HISTORY CLEANER"
echo "====================================="
echo ""

# Warning
echo -e "${RED}⚠️  WARNING: This will rewrite git history!${NC}"
echo ""
echo "This will:"
echo "  1. Remove all exposed secrets from git history"
echo "  2. Force push to remote (requires coordination with team)"
echo "  3. All collaborators will need to re-clone the repository"
echo ""
echo -n "Do you want to continue? (yes/no): "
read -r CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Not in a git repository!${NC}"
    exit 1
fi

# Create backup
echo "📦 Creating backup..."
BACKUP_DIR="../ai-stanbul-backup-$(date +%Y%m%d_%H%M%S)"
cp -r . "$BACKUP_DIR"
echo -e "${GREEN}✅ Backup created at: $BACKUP_DIR${NC}"
echo ""

# Check for BFG
if command -v bfg &> /dev/null; then
    echo "🔧 Using BFG Repo-Cleaner (recommended)..."
    
    # Create passwords file
    cat > /tmp/ai-istanbul-secrets.txt <<EOF
*iwP#MDmX5dn8V:1LExE|70:O>|i
AIzaSyDiQjBfo7Lk9WOL7ut4wbiNbNWQpgr1k9Q
49575391e412bd4332062ffdb688c38c
Ozw5vFR0HzgXPPtNk1DdZwCfRL7Dl6HwGe_m0CN_zfg
%2AiwP%23MDmX5dn8V%3A1LExE%7C70%3AO%3E%7Ci
EOF
    
    # Run BFG
    bfg --replace-text /tmp/ai-istanbul-secrets.txt .
    
    # Cleanup
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    
    # Remove temp file
    rm /tmp/ai-istanbul-secrets.txt
    
    echo -e "${GREEN}✅ Secrets removed from history${NC}"
    
elif command -v git-filter-repo &> /dev/null; then
    echo "🔧 Using git-filter-repo..."
    
    # Create replace file
    cat > /tmp/ai-istanbul-replacements.txt <<EOF
*iwP#MDmX5dn8V:1LExE|70:O>|i==>REDACTED_PASSWORD
AIzaSyDiQjBfo7Lk9WOL7ut4wbiNbNWQpgr1k9Q==>REDACTED_GOOGLE_API_KEY
49575391e412bd4332062ffdb688c38c==>REDACTED_OPENWEATHER_KEY
Ozw5vFR0HzgXPPtNk1DdZwCfRL7Dl6HwGe_m0CN_zfg==>REDACTED_SECRET_KEY
%2AiwP%23MDmX5dn8V%3A1LExE%7C70%3AO%3E%7Ci==>REDACTED_ENCODED_PASSWORD
EOF
    
    # Run git-filter-repo
    git-filter-repo --replace-text /tmp/ai-istanbul-replacements.txt --force
    
    # Remove temp file
    rm /tmp/ai-istanbul-replacements.txt
    
    echo -e "${GREEN}✅ Secrets removed from history${NC}"
    
else
    echo -e "${YELLOW}⚠️  Neither BFG nor git-filter-repo found${NC}"
    echo ""
    echo "Please install one of them:"
    echo "  brew install bfg"
    echo "  brew install git-filter-repo"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ GIT HISTORY CLEANED!"
echo "=============================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Review the changes:"
echo "   git log --oneline -10"
echo ""
echo "2. Force push to remote (⚠️ DANGEROUS!):"
echo "   git push origin --force --all"
echo "   git push origin --force --tags"
echo ""
echo "3. Notify all collaborators to:"
echo "   - Delete their local copy"
echo "   - Re-clone from GitHub"
echo ""
echo -e "${YELLOW}⚠️  Important:${NC}"
echo "  - Backup is at: $BACKUP_DIR"
echo "  - Verify the changes before force pushing"
echo "  - Force push will affect all collaborators"
echo ""
echo -n "Do you want to force push now? (yes/no): "
read -r PUSH_CONFIRM

if [ "$PUSH_CONFIRM" = "yes" ]; then
    echo ""
    echo "🚀 Force pushing to remote..."
    
    # Get current branch
    CURRENT_BRANCH=$(git branch --show-current)
    
    # Force push
    git push origin --force "$CURRENT_BRANCH"
    
    echo -e "${GREEN}✅ Force push complete!${NC}"
    echo ""
    echo "🎉 Your repository is now clean!"
    echo ""
    echo "📧 Remember to:"
    echo "  1. Close the GitGuardian alert"
    echo "  2. Notify team members about the force push"
    echo "  3. Verify the application still works"
else
    echo ""
    echo "Skipping force push."
    echo "When ready, run:"
    echo "  git push origin --force --all"
    echo "  git push origin --force --tags"
fi

echo ""
echo "✅ Done!"
