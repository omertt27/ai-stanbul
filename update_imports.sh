#!/bin/bash
# Update all imports from old core.main_system to new main_system

echo "🔄 Updating all imports to use unified main_system.py..."

# Find all Python files with the old import (excluding backups and __pycache__)
FILES=$(grep -rl "from istanbul_ai.core.main_system import" --include="*.py" . | grep -v ".backup" | grep -v "__pycache__")

COUNT=0
for file in $FILES; do
    echo "  Updating: $file"
    # Use sed to replace the import (macOS compatible)
    sed -i '' 's|from istanbul_ai.core.main_system import|from istanbul_ai.main_system import|g' "$file"
    COUNT=$((COUNT + 1))
done

echo "✅ Updated $COUNT files"
echo ""
echo "📊 Verification:"
grep -r "from istanbul_ai.main_system import IstanbulDailyTalkAI" --include="*.py" . | grep -v ".backup" | grep -v "__pycache__" | wc -l | xargs echo "  Files with new import:"
grep -r "from istanbul_ai.core.main_system import" --include="*.py" . | grep -v ".backup" | grep -v "__pycache__" | wc -l | xargs echo "  Files with old import:"
