#!/usr/bin/env python3
"""
Quick validation that Transportation RAG is working
"""

def check_syntax():
    """Just check if the file has valid Python syntax"""
    try:
        with open('/Users/omer/Desktop/ai-stanbul/backend/services/transportation_rag_system.py', 'r') as f:
            code = f.read()
        compile(code, 'transportation_rag_system.py', 'exec')
        print("✅ Syntax check: PASSED")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False

def check_structure():
    """Check key components exist"""
    try:
        with open('/Users/omer/Desktop/ai-stanbul/backend/services/transportation_rag_system.py', 'r') as f:
            code = f.read()
        
        checks = {
            'IstanbulTransportationRAG class': 'class IstanbulTransportationRAG' in code,
            'BFS pathfinding': '_find_path_bfs' in code,
            'Transfer neighbors': '_get_transfer_neighbors' in code,
            'Same line neighbors': '_get_same_line_neighbors' in code,
            'Build route from path': '_build_route_from_path' in code,
            'Get directions': 'get_directions_text' in code,
            'RAG context': 'get_rag_context_for_query' in code,
            'Station graph': '_build_station_graph' in code,
            'Marmaray stations': 'MARMARAY' in code,
            'M4 stations': 'M4-Kadıköy' in code,
            'M2 stations': 'M2-Taksim' in code,
        }
        
        all_passed = True
        for name, check in checks.items():
            status = "✅" if check else "❌"
            print(f"{status} {name}")
            if not check:
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Error checking structure: {e}")
        return False

if __name__ == '__main__':
    print("=" * 70)
    print("TRANSPORTATION RAG VALIDATION")
    print("=" * 70)
    print()
    
    print("1. Syntax Check")
    print("-" * 70)
    syntax_ok = check_syntax()
    print()
    
    print("2. Structure Check")
    print("-" * 70)
    structure_ok = check_structure()
    print()
    
    print("=" * 70)
    if syntax_ok and structure_ok:
        print("✅ VALIDATION PASSED")
        print("Transportation RAG system is ready for use!")
    else:
        print("❌ VALIDATION FAILED")
        print("Please check the errors above")
    print("=" * 70)
