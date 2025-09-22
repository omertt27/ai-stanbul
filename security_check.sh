#!/bin/bash
# AI Istanbul - Security Validation Script
# ==============================================
# This script checks for exposed sensitive files and configurations

echo "🔒 AI Istanbul Security Validation Script"
echo "=========================================="
echo "Checking for exposed sensitive files..."
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Initialize counters
ISSUES_FOUND=0
FILES_CHECKED=0

# Function to report issues
report_issue() {
    echo -e "${RED}❌ SECURITY ISSUE: $1${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

# Function to report success
report_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to report warning
report_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

echo "1. Checking for exposed environment files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
if [ -f ".env" ] && git ls-files --error-unmatch .env >/dev/null 2>&1; then
    report_issue ".env file is tracked in git!"
elif [ -f ".env" ]; then
    report_success ".env file exists but is not tracked"
else
    report_warning ".env file not found"
fi

if [ -f "backend/.env" ] && git ls-files --error-unmatch backend/.env >/dev/null 2>&1; then
    report_issue "backend/.env file is tracked in git!"
elif [ -f "backend/.env" ]; then
    report_success "backend/.env file exists but is not tracked"
else
    report_warning "backend/.env file not found"
fi

echo
echo "2. Checking for exposed database files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
for db_file in *.db *.sqlite *.sqlite3 backend/*.db; do
    if [ -f "$db_file" ] && git ls-files --error-unmatch "$db_file" >/dev/null 2>&1; then
        report_issue "Database file $db_file is tracked in git!"
    elif [ -f "$db_file" ]; then
        report_success "Database file $db_file exists but is not tracked"
    fi
done

echo
echo "3. Checking for exposed credential files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
for cred_dir in credentials backend/credentials; do
    if [ -d "$cred_dir" ] && git ls-files "$cred_dir" | grep -q .; then
        report_issue "Credential directory $cred_dir contains tracked files!"
    elif [ -d "$cred_dir" ]; then
        report_success "Credential directory $cred_dir exists but is not tracked"
    fi
done

echo
echo "4. Checking for exposed log files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
for log_file in *.log backend/*.log ai_istanbul.log; do
    if [ -f "$log_file" ] && git ls-files --error-unmatch "$log_file" >/dev/null 2>&1; then
        report_issue "Log file $log_file is tracked in git!"
    elif [ -f "$log_file" ]; then
        report_success "Log file $log_file exists but is not tracked"
    fi
done

echo
echo "5. Checking for API keys in tracked files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
if git grep -i "api_key\|apikey\|secret_key\|password\|token" -- '*.js' '*.jsx' '*.py' '*.md' 2>/dev/null | grep -v "example\|template\|placeholder\|<your" | grep -q .; then
    report_issue "Potential API keys or secrets found in tracked files!"
    echo "   Run: git grep -i \"api_key\\|apikey\\|secret_key\\|password\\|token\" -- '*.js' '*.jsx' '*.py' '*.md'"
else
    report_success "No obvious API keys found in tracked files"
fi

echo
echo "6. Checking .gitignore coverage..."
FILES_CHECKED=$((FILES_CHECKED + 1))
if [ -f ".gitignore" ]; then
    if grep -q "\.env" .gitignore; then
        report_success ".gitignore covers .env files"
    else
        report_issue ".gitignore doesn't cover .env files!"
    fi
    
    if grep -q "\*\.db\|\*\.sqlite" .gitignore; then
        report_success ".gitignore covers database files"
    else
        report_issue ".gitignore doesn't cover database files!"
    fi
    
    if grep -q "\*\.log" .gitignore; then
        report_success ".gitignore covers log files"
    else
        report_issue ".gitignore doesn't cover log files!"
    fi
else
    report_issue ".gitignore file missing!"
fi

echo
echo "7. Checking for build files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
for build_dir in frontend/dist frontend/build backend/build; do
    if [ -d "$build_dir" ] && git ls-files "$build_dir" | grep -q .; then
        report_issue "Build directory $build_dir contains tracked files!"
    elif [ -d "$build_dir" ]; then
        report_success "Build directory $build_dir exists but is not tracked"
    fi
done

echo
echo "8. Checking for source maps..."
FILES_CHECKED=$((FILES_CHECKED + 1))
if git ls-files | grep -q "\.map$"; then
    report_issue "Source map files are tracked in git!"
else
    report_success "No source map files tracked"
fi

echo
echo "9. Checking frontend protection files..."
FILES_CHECKED=$((FILES_CHECKED + 1))
if [ -f "frontend/src/utils/websiteProtection.js" ]; then
    report_success "Website protection script exists"
else
    report_warning "Website protection script missing"
fi

if [ -f "frontend/src/styles/anti-copy.css" ]; then
    report_success "Anti-copy CSS exists"
else
    report_warning "Anti-copy CSS missing"
fi

if [ -f "frontend/src/pages/TermsOfService.jsx" ]; then
    report_success "Terms of Service page exists"
else
    report_warning "Terms of Service page missing"
fi

echo
echo "=========================================="
echo "Security Validation Complete"
echo "Files checked: $FILES_CHECKED"

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}🎉 No security issues found!${NC}"
    exit 0
else
    echo -e "${RED}❌ $ISSUES_FOUND security issues found!${NC}"
    echo
    echo "Recommended actions:"
    echo "1. Review and fix all reported issues"
    echo "2. Rotate any exposed API keys"
    echo "3. Update .gitignore if needed"
    echo "4. Run: git rm --cached <file> to untrack sensitive files"
    echo "5. Consider using git filter-branch to remove from history"
    exit 1
fi
