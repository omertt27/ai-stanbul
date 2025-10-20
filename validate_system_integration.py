#!/usr/bin/env python3
"""
System Integration Validator
Validates which museum and attractions systems are actually integrated into main_system.py
"""

import sys
import os
import importlib.util

def check_module_exists(module_path, module_name):
    """Check if a module file exists"""
    if os.path.exists(module_path):
        print(f"✅ {module_name} exists at: {module_path}")
        return True
    else:
        print(f"❌ {module_name} NOT FOUND at: {module_path}")
        return False

def check_import_in_main_system(import_name):
    """Check if a class/module is imported in main_system.py"""
    main_system_path = "istanbul_ai/main_system.py"
    
    if not os.path.exists(main_system_path):
        print(f"❌ main_system.py not found at {main_system_path}")
        return False
    
    with open(main_system_path, 'r') as f:
        content = f.read()
        
    if import_name in content:
        # Count occurrences
        count = content.count(import_name)
        print(f"✅ '{import_name}' found {count} times in main_system.py")
        
        # Find import statements
        import_lines = [line.strip() for line in content.split('\n') if 'import' in line and import_name in line]
        if import_lines:
            print(f"   Import statements:")
            for line in import_lines[:3]:  # Show first 3
                print(f"      {line}")
        return True
    else:
        print(f"❌ '{import_name}' NOT found in main_system.py")
        return False

def check_class_usage_in_main_system(class_name):
    """Check if a class is actually instantiated in main_system.py"""
    main_system_path = "istanbul_ai/main_system.py"
    
    if not os.path.exists(main_system_path):
        return False
    
    with open(main_system_path, 'r') as f:
        content = f.read()
    
    # Look for instantiation patterns
    patterns = [
        f"self.{class_name.lower()}",
        f"{class_name}()",
        f"= {class_name}("
    ]
    
    for pattern in patterns:
        if pattern in content:
            print(f"✅ {class_name} appears to be instantiated (found pattern: '{pattern}')")
            return True
    
    print(f"❌ {class_name} does NOT appear to be instantiated in main_system.py")
    return False

def main():
    """Run validation checks"""
    print("=" * 80)
    print("ISTANBUL AI SYSTEM INTEGRATION VALIDATOR")
    print("=" * 80)
    print()
    
    # Check if required modules exist
    print("📋 STEP 1: Checking if system files exist...")
    print("-" * 80)
    
    modules_to_check = {
        'MuseumResponseGenerator': 'museum_response_generator.py',
        'IstanbulMuseumSystem': 'museum_advising_system.py',
        'IstanbulAttractionsSystem': 'istanbul_attractions_system.py',
        'MainSystem': 'istanbul_ai/main_system.py'
    }
    
    existence_results = {}
    for name, path in modules_to_check.items():
        existence_results[name] = check_module_exists(path, name)
    
    print()
    
    # Check imports in main_system.py
    print("📋 STEP 2: Checking imports in main_system.py...")
    print("-" * 80)
    
    import_results = {}
    imports_to_check = [
        'MuseumResponseGenerator',
        'IstanbulMuseumSystem', 
        'IstanbulAttractionsSystem',
        'museum_response_generator',
        'museum_advising_system',
        'istanbul_attractions_system'
    ]
    
    for import_name in imports_to_check:
        import_results[import_name] = check_import_in_main_system(import_name)
    
    print()
    
    # Check class instantiation
    print("📋 STEP 3: Checking class instantiation in main_system.py...")
    print("-" * 80)
    
    instantiation_results = {}
    classes_to_check = [
        'MuseumResponseGenerator',
        'IstanbulMuseumSystem',
        'IstanbulAttractionsSystem'
    ]
    
    for class_name in classes_to_check:
        instantiation_results[class_name] = check_class_usage_in_main_system(class_name)
    
    print()
    
    # Summary
    print("=" * 80)
    print("📊 INTEGRATION STATUS SUMMARY")
    print("=" * 80)
    print()
    
    print("✅ INTEGRATED SYSTEMS:")
    integrated_count = 0
    for class_name in classes_to_check:
        if existence_results.get(class_name, False) and \
           import_results.get(class_name, False) and \
           instantiation_results.get(class_name, False):
            print(f"   • {class_name}")
            integrated_count += 1
    
    if integrated_count == 0:
        print("   (None - only MuseumResponseGenerator might be integrated)")
    
    print()
    print("❌ AVAILABLE BUT NOT INTEGRATED:")
    not_integrated_count = 0
    for class_name in classes_to_check:
        if existence_results.get(class_name, False) and \
           not (import_results.get(class_name, False) and instantiation_results.get(class_name, False)):
            print(f"   • {class_name}")
            not_integrated_count += 1
    
    if not_integrated_count == 0:
        print("   (None)")
    
    print()
    print("=" * 80)
    print("🎯 CONCLUSIONS:")
    print("=" * 80)
    
    if not_integrated_count > 0:
        print(f"⚠️  Found {not_integrated_count} advanced system(s) that exist but are NOT integrated!")
        print(f"   These systems have full feature support but are never called by the main system.")
        print(f"   This explains the low feature coverage in test results.")
        print()
        print("💡 RECOMMENDATION: Integrate these systems to dramatically improve feature coverage!")
    else:
        print("✅ All available systems appear to be integrated!")
    
    print()
    
    # Check for common museum/attraction query handling
    print("=" * 80)
    print("🔍 QUERY ROUTING ANALYSIS")
    print("=" * 80)
    
    main_system_path = "istanbul_ai/main_system.py"
    if os.path.exists(main_system_path):
        with open(main_system_path, 'r') as f:
            content = f.read()
        
        print()
        print("Checking how museum queries are routed...")
        
        if 'museum_generator.generate_museum_recommendation' in content:
            print("✅ Routes to: MuseumResponseGenerator.generate_museum_recommendation()")
            print("   Type: Pre-written narratives (limited features)")
        
        if 'IstanbulMuseumSystem' in content and 'process_museum_query' in content:
            print("✅ Routes to: IstanbulMuseumSystem.process_museum_query()")
            print("   Type: Advanced ML system with GPS, categories, etc.")
        elif 'IstanbulMuseumSystem' not in content:
            print("❌ IstanbulMuseumSystem NOT integrated - no advanced museum features!")
        
        print()
        print("Checking how attraction queries are routed...")
        
        if 'response_generator.generate_comprehensive_recommendation' in content:
            print("✅ Routes to: ResponseGenerator.generate_comprehensive_recommendation()")
            print("   Type: Generic responses (basic features)")
        
        if 'IstanbulAttractionsSystem' in content:
            print("✅ Routes to: IstanbulAttractionsSystem")
            print("   Type: Advanced attractions system with 78+ curated attractions")
        else:
            print("❌ IstanbulAttractionsSystem NOT integrated - limited attraction features!")
    
    print()
    print("=" * 80)
    print("Validation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
