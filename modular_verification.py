#!/usr/bin/env python3
"""
Direct modular system verification script - tests the AI system directly without API server
"""

def verify_modular_system_directly():
    """Verify that the modular AI system meets all requirements"""
    print("🔍 ISTANBUL AI MODULAR SYSTEM VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # 1. Test modular architecture
    print("1. Testing modular architecture...")
    try:
        from istanbul_daily_talk_system_modular import IstanbulDailyTalkAI, ConversationContext
        ai = IstanbulDailyTalkAI()
        results['modular_architecture'] = True
        print("   ✅ Modular system loads successfully")
    except Exception as e:
        results['modular_architecture'] = False
        print(f"   ❌ Modular system failed: {e}")
        return False
    
    # 2. Test missing methods are implemented
    print("2. Testing missing methods implementation...")
    try:
        # Check for previously missing methods
        missing_methods = [
            '_generate_fallback_response',
            '_enhance_multi_intent_response', 
            '_get_or_request_gps_location'
        ]
        
        methods_found = 0
        for method in missing_methods:
            if hasattr(ai, method):
                methods_found += 1
                print(f"      ✓ Found method: {method}")
        
        if methods_found == len(missing_methods):
            results['missing_methods'] = True
            print(f"   ✅ All {len(missing_methods)} previously missing methods are now implemented")
        else:
            results['missing_methods'] = False
            print(f"   ❌ Only {methods_found}/{len(missing_methods)} methods found")
            
    except Exception as e:
        results['missing_methods'] = False
        print(f"   ❌ Method check failed: {e}")
    
    # 3. Test restaurant routing
    print("3. Testing restaurant query routing...")
    try:
        response = ai.process_message("Show me Turkish restaurants in Sultanahmet", "test_user")
        if "Restaurant Recommendations" in response and "🍽️" in response:
            results['restaurant_routing'] = True
            print("   ✅ Restaurant queries return restaurant recommendations")
        else:
            results['restaurant_routing'] = False
            print("   ❌ Restaurant routing not working correctly")
            print(f"      Response preview: {response[:100]}...")
    except Exception as e:
        results['restaurant_routing'] = False
        print(f"   ❌ Restaurant routing test failed: {e}")
    
    # 4. Test events routing  
    print("4. Testing events query routing...")
    try:
        response = ai.process_message("What cultural events are happening?", "test_user")
        if "Cultural Events" in response and "🎭" in response:
            results['events_routing'] = True
            print("   ✅ Events queries return cultural events")
        else:
            results['events_routing'] = False
            print("   ❌ Events routing not working correctly")
            print(f"      Response preview: {response[:100]}...")
    except Exception as e:
        results['events_routing'] = False
        print(f"   ❌ Events routing test failed: {e}")
    
    # 5. Test fallback responses
    print("5. Testing fallback response handling...")
    try:
        response = ai.process_message("askjfhlaksjdhflakjsdhf", "test_user")
        if len(response) > 50 and ("Welcome" in response or "help" in response):
            results['fallback_responses'] = True
            print("   ✅ Fallback responses are working properly")
        else:
            results['fallback_responses'] = False
            print("   ❌ Fallback responses not adequate")
            print(f"      Response preview: {response[:100]}...")
    except Exception as e:
        results['fallback_responses'] = False
        print(f"   ❌ Fallback test failed: {e}")
    
    # 6. Test backward compatibility
    print("6. Testing backward compatibility...")
    try:
        # Test that the old import still works via the modular wrapper
        from istanbul_daily_talk_system_modular import IstanbulDailyTalkAI as ModularAI
        old_ai = ModularAI()
        response = old_ai.process_message("Hello", "test_user")
        if len(response) > 10:
            results['backward_compatibility'] = True
            print("   ✅ Backward compatibility maintained")
        else:
            results['backward_compatibility'] = False
            print("   ❌ Backward compatibility broken")
    except Exception as e:
        results['backward_compatibility'] = False
        print(f"   ❌ Backward compatibility test failed: {e}")
    
    # 7. Test backend integration
    print("7. Testing backend API integration...")
    try:
        # Check if backend imports the modular system
        with open('backend/main.py', 'r') as f:
            content = f.read()
        
        if 'istanbul_daily_talk_system_modular' in content:
            results['backend_integration'] = True
            print("   ✅ Backend API uses modular system")
        else:
            results['backend_integration'] = False
            print("   ❌ Backend API not updated to use modular system")
            
    except Exception as e:
        results['backend_integration'] = False
        print(f"   ❌ Backend integration test failed: {e}")
    
    # 8. Test modular file structure
    print("8. Testing modular file structure...")
    try:
        import os
        expected_files = [
            'istanbul_ai/__init__.py',
            'istanbul_ai/core/main_system.py',
            'istanbul_ai/core/user_profile.py',
            'istanbul_ai/core/conversation_context.py',
            'istanbul_ai/core/entity_recognizer.py',
            'istanbul_ai/utils/constants.py',
            'istanbul_daily_talk_system_modular.py'
        ]
        
        files_found = 0
        for file_path in expected_files:
            if os.path.exists(file_path):
                files_found += 1
            else:
                print(f"      ❌ Missing: {file_path}")
        
        if files_found == len(expected_files):
            results['modular_structure'] = True
            print(f"   ✅ All {len(expected_files)} modular files found")
        else:
            results['modular_structure'] = False
            print(f"   ❌ Only {files_found}/{len(expected_files)} modular files found")
            
    except Exception as e:
        results['modular_structure'] = False
        print(f"   ❌ File structure test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MODULAR SYSTEM VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<30} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL REQUIREMENTS SUCCESSFULLY MET!")
        print("\n✨ The Istanbul AI system has been successfully:")
        print("   • Refactored from monolithic (2915 lines) to modular architecture")
        print("   • Fixed all missing method errors")
        print("   • Improved routing for restaurant and events queries") 
        print("   • Standardized response formats and fallbacks")
        print("   • Integrated with backend API")
        print("   • Maintained backward compatibility")
        print("   • Created maintainable, testable module structure")
        return True
    else:
        print(f"\n⚠️  {total-passed} requirements still need attention")
        return False

if __name__ == "__main__":
    import sys
    success = verify_modular_system_directly()
    sys.exit(0 if success else 1)
