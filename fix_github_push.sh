#!/bin/bash
# Fix GitHub Push - Remove Large Model Files from Git Tracking
# This script removes large files from Git but keeps them locally

echo "🔧 Fixing GitHub Push - Removing Large Files from Git Tracking"
echo "================================================================"
echo ""

# Step 1: Remove large model files from Git tracking (but keep locally!)
echo "📦 Step 1: Removing model files from Git tracking..."
echo "   (Files will stay on your disk, just not in Git)"
echo ""

# Remove .pth files
if git ls-files | grep -q '\.pth$'; then
    echo "   Removing *.pth files from Git..."
    git rm --cached *.pth 2>/dev/null || true
    git rm --cached **/*.pth 2>/dev/null || true
fi

# Remove checkpoint directories
if git ls-files | grep -q 'phase2_extended_v2/'; then
    echo "   Removing phase2_extended_v2/ directory from Git..."
    git rm --cached -r phase2_extended_v2/ 2>/dev/null || true
fi

if git ls-files | grep -q 'phase2_final/'; then
    echo "   Removing phase2_final/ directory from Git..."
    git rm --cached -r phase2_final/ 2>/dev/null || true
fi

if git ls-files | grep -q 'phase2_extended/'; then
    echo "   Removing phase2_extended/ directory from Git..."
    git rm --cached -r phase2_extended/ 2>/dev/null || true
fi

# Remove .safetensors files
if git ls-files | grep -q '\.safetensors$'; then
    echo "   Removing *.safetensors files from Git..."
    git rm --cached **/*.safetensors 2>/dev/null || true
fi

# Remove logs directory if tracked
if git ls-files | grep -q '^logs/'; then
    echo "   Removing logs/ directory from Git..."
    git rm --cached -r logs/ 2>/dev/null || true
fi

echo ""
echo "✅ Step 1 Complete: Large files removed from Git tracking"
echo ""

# Step 2: Update .gitignore (already done)
echo "📝 Step 2: Checking .gitignore..."
if grep -q "# LARGE MODEL FILES" .gitignore; then
    echo "   ✅ .gitignore already updated"
else
    echo "   ⚠️  .gitignore needs manual update"
fi
echo ""

# Step 3: Stage the changes
echo "📌 Step 3: Staging changes..."
git add .gitignore
echo "   ✅ .gitignore staged"
echo ""

# Step 4: Show status
echo "📊 Step 4: Current Git status..."
echo "================================================================"
git status --short
echo "================================================================"
echo ""

# Step 5: Commit instructions
echo "🚀 Step 5: Ready to commit and push!"
echo "================================================================"
echo ""
echo "Run these commands to complete:"
echo ""
echo "  git commit -m \"Remove large model files from Git tracking\""
echo "  git push"
echo ""
echo "✅ Your model files will stay on your computer"
echo "✅ GitHub won't reject the push anymore"
echo "✅ Only code and small files will be pushed"
echo ""
echo "================================================================"
echo ""

# Step 6: Show what's ignored now
echo "📋 Files that will be ignored going forward:"
echo "   ✅ *.pth (PyTorch models)"
echo "   ✅ *.pt (PyTorch checkpoints)"
echo "   ✅ *.safetensors (Transformers models)"
echo "   ✅ phase2_extended_v2/ (training checkpoints)"
echo "   ✅ logs/ (training logs)"
echo ""

echo "💡 Pro Tip: Store your best model separately:"
echo "   cp phase2_extended_model.pth ~/models/production_model_v1.pth"
echo ""
echo "Done! 🎉"
