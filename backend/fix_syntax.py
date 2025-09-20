#!/usr/bin/env python3
"""Fix syntax errors in main.py"""

import re

def fix_main_py():
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Fix the f-string syntax error on line 1849
    content = re.sub(
        r'print\(f"Response guidance error: \{e\}"\)',
        'print(f"Response guidance error: {e}")',
        content
    )
    
    # Write back the fixed content
    with open('main.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed f-string syntax error in main.py")

if __name__ == "__main__":
    fix_main_py()
