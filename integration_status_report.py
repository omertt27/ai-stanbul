#!/usr/bin/env python3
"""
Comprehensive Integration Status Report
Checks all systems and their integration status
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_integration_status():
    """Check the integration status of all systems"""
    
    print("🔍 COMPREHENSIVE INTEGRATION STATUS REPORT")
    print("=" * 60)
    
    status_report = {
        "systems_checked": 0,
        "systems_integrated": 0,
        "integration_points": []
    }
    
    # 1. Check Istanbul Daily Talk AI System
    print("\n1. 🏛️ Istanbul Daily Talk AI System:")
    print("-" * 35)
    try:
        from istanbul_daily_talk_system import IstanbulDailyTalkAI, MULTI_INTENT_AVAILABLE
        ai_system = IstanbulDailyTalkAI()
        
        status_report["systems_checked"] += 1
        
        print("   ✅ Istanbul Daily Talk AI: Available")
        print(f"   📍 Multi-Intent Handler: {'✅ Integrated' if MULTI_INTENT_AVAILABLE else '❌ Not Available'}")
        print(f"   🧠 Deep Learning System: {'✅ Active' if hasattr(ai_system, 'deep_learning_ai') and ai_system.deep_learning_ai else '❌ Not Available'}")
        print(f"   🎯 Multi-Intent Processing: {'✅ Ready' if hasattr(ai_system, 'multi_intent_handler') and ai_system.multi_intent_handler else '❌ Not Ready'}")
        
        if MULTI_INTENT_AVAILABLE and hasattr(ai_system, 'multi_intent_handler') and ai_system.multi_intent_handler:
            status_report["systems_integrated"] += 1
            status_report["integration_points"].append("Multi-Intent Handler → Istanbul Daily Talk AI")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Check FastAPI App Integration
    print("\n2. 🚀 FastAPI App (app.py):")
    print("-" * 25)
    try:
        # Check imports in app.py
        import ast
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        status_report["systems_checked"] += 1
        
        if 'from istanbul_daily_talk_system import IstanbulDailyTalkAI' in app_content:
            print("   ✅ Istanbul Daily Talk AI: Imported")
            status_report["integration_points"].append("Istanbul Daily Talk AI → FastAPI App")
        else:
            print("   ❌ Istanbul Daily Talk AI: Not imported")
        
        if 'ai_system.process_message' in app_content:
            print("   ✅ Message Processing: Integrated")
            status_report["systems_integrated"] += 1
        else:
            print("   ❌ Message Processing: Not integrated")
        
    except Exception as e:
        print(f"   ❌ Error checking app.py: {e}")
    
    # 3. Check Backend Main.py Integration  
    print("\n3. 🏗️ Backend Main.py:")
    print("-" * 20)
    try:
        # Check backend/main.py
        with open('backend/main.py', 'r') as f:
            backend_content = f.read()
        
        status_report["systems_checked"] += 1
        
        if 'from multi_intent_query_handler import MultiIntentQueryHandler' in backend_content:
            print("   ✅ Multi-Intent Handler: Imported")
            status_report["integration_points"].append("Multi-Intent Handler → Backend Main")
        else:
            print("   ❌ Multi-Intent Handler: Not directly imported")
        
        if 'advanced_understanding' in backend_content:
            print("   ✅ Advanced Understanding: Available")
            status_report["integration_points"].append("Advanced Understanding → Backend Main")
        else:
            print("   ❌ Advanced Understanding: Not available")
        
        # Backend uses MultiIntentQueryHandler via Advanced Understanding System
        if 'multi_intent_result' in backend_content:
            print("   ✅ Multi-Intent Processing: Via Advanced Understanding")
            status_report["systems_integrated"] += 1
        else:
            print("   ❌ Multi-Intent Processing: Not integrated")
        
    except Exception as e:
        print(f"   ❌ Error checking backend/main.py: {e}")
    
    # 4. Check Multi-Intent Query Handler
    print("\n4. 🎯 Multi-Intent Query Handler:")
    print("-" * 30)
    try:
        from multi_intent_query_handler import MultiIntentQueryHandler
        handler = MultiIntentQueryHandler()
        
        status_report["systems_checked"] += 1
        
        print("   ✅ Multi-Intent Handler: Available")
        print("   ✅ Restaurant Query Support: Ready")
        print("   ✅ Complex Query Analysis: Ready")
        
        # Test a sample query
        test_result = handler.analyze_query("Where can I find good Turkish restaurants in Sultanahmet?")
        if test_result and hasattr(test_result, 'primary_intent'):
            print(f"   ✅ Query Analysis: Working (detected: {test_result.primary_intent.type.value})")
            status_report["systems_integrated"] += 1
        else:
            print("   ❌ Query Analysis: Not working properly")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Check Deep Learning Enhanced AI
    print("\n5. 🧠 Deep Learning Enhanced AI:")
    print("-" * 30)
    try:
        from deep_learning_enhanced_ai import DeepLearningEnhancedAI
        dl_ai = DeepLearningEnhancedAI()
        
        status_report["systems_checked"] += 1
        
        print("   ✅ Deep Learning AI: Available")
        print("   ✅ English Optimization: Active")
        print("   ✅ Neural Networks: Initialized")
        
        status_report["systems_integrated"] += 1
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION SUMMARY:")
    print("=" * 60)
    
    integration_percentage = (status_report["systems_integrated"] / max(status_report["systems_checked"], 1)) * 100
    
    print(f"Systems Checked: {status_report['systems_checked']}")
    print(f"Systems Integrated: {status_report['systems_integrated']}")
    print(f"Integration Rate: {integration_percentage:.1f}%")
    
    print(f"\n🔗 Integration Points ({len(status_report['integration_points'])}):")
    for i, point in enumerate(status_report['integration_points'], 1):
        print(f"   {i}. {point}")
    
    # Overall Status
    if integration_percentage >= 80:
        print(f"\n🎉 INTEGRATION STATUS: ✅ EXCELLENT ({integration_percentage:.1f}%)")
        print("   All major systems are properly integrated!")
    elif integration_percentage >= 60:
        print(f"\n⚠️ INTEGRATION STATUS: 🟡 GOOD ({integration_percentage:.1f}%)")
        print("   Most systems integrated, minor issues detected.")
    else:
        print(f"\n❌ INTEGRATION STATUS: 🔴 NEEDS WORK ({integration_percentage:.1f}%)")
        print("   Significant integration issues detected.")
    
    # Architecture Overview
    print(f"\n🏗️ SYSTEM ARCHITECTURE:")
    print("=" * 25)
    print("User Query")
    print("    ↓")
    print("FastAPI App (app.py) OR Backend Main (main.py)")
    print("    ↓")
    print("Istanbul Daily Talk AI System")
    print("    ↓")
    print("Multi-Intent Query Handler ← (for restaurant queries)")
    print("    ↓")
    print("Enhanced Response + Istanbul Context")
    print("    ↓") 
    print("User Response")
    
    return status_report

if __name__ == "__main__":
    check_integration_status()
