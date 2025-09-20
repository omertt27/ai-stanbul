#!/usr/bin/env python3
"""
Comprehensive syntax fix for dictionary and f-string issues in main.py
"""

import re

def fix_dictionary_syntax():
    """Fix all dictionary syntax issues"""
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Fix patterns like: {"key": f"value {var}") -> {"key": f"value {var}"}
    content = re.sub(r'(\{"[^"]*":\s*f"[^"]*\{[^}]*\}[^"]*")\)', r'\1}', content)
    
    # Fix patterns like: {"key": "value") -> {"key": "value"}
    content = re.sub(r'(\{"[^"]*":\s*"[^"]*")\)', r'\1}', content)
    
    # Fix patterns like: return {"key": f"value {var}") -> return {"key": f"value {var}"}
    content = re.sub(r'(return\s*\{"[^"]*":\s*f"[^"]*\{[^}]*\}[^"]*")\)', r'\1}', content)
    
    # Fix patterns like: messages.append({"role": "system", "content": f"text {var}")) 
    # -> messages.append({"role": "system", "content": f"text {var}"})
    content = re.sub(r'(messages\.append\(\{"[^"]*":\s*"[^"]*",\s*"[^"]*":\s*f"[^"]*\{[^}]*\}[^"]*"\))\)', r'\1', content)
    
    with open('main.py', 'w') as f:
        f.write(content)
    
    print("Fixed dictionary syntax issues")

if __name__ == "__main__":
    fix_dictionary_syntax()
