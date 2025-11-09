#!/usr/bin/env python3
"""
Comprehensive Script to Integrate Enhanced LLM Client into All Handlers

This script updates all Istanbul AI handlers to use the Google Cloud Llama 3.1 8B LLM
via the enhanced LLM client with comprehensive features:
- Typo correction
- Multilingual support (automatic detection)
- Weather awareness
- GPS integration
- Context-aware responses

Author: Istanbul AI Team
Date: [Current Date]
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Handler files to update
HANDLERS = [
    "istanbul_ai/handlers/neighborhood_handler.py",
    "istanbul_ai/handlers/transportation_handler.py",
    "istanbul_ai/handlers/weather_handler.py",
    "istanbul_ai/handlers/event_handler.py",
    "istanbul_ai/handlers/hidden_gems_handler.py",
    "istanbul_ai/handlers/local_food_handler.py"
]

# Import statement to add
LLM_IMPORT = """
# Import enhanced LLM client
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from enhanced_llm_config import get_enhanced_llm_client, EnhancedLLMClient
    ENHANCED_LLM_AVAILABLE = True
except ImportError as e:
    ENHANCED_LLM_AVAILABLE = False
    logging.warning(f"⚠️ Enhanced LLM client not available: {e}")
"""

# Initialization code to add to __init__ methods
INIT_CODE = """
        # Initialize enhanced LLM client
        if ENHANCED_LLM_AVAILABLE:
            try:
                self.llm_client = get_enhanced_llm_client()
                self.has_enhanced_llm = True
                logger.info("✅ Enhanced LLM client (Google Cloud Llama 3.1 8B) initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize enhanced LLM client: {e}")
                self.llm_client = None
                self.has_enhanced_llm = False
        else:
            self.llm_client = None
            self.has_enhanced_llm = False
"""


def check_file_exists(filepath):
    """Check if handler file exists"""
    if not os.path.exists(filepath):
        logger.error(f"❌ File not found: {filepath}")
        return False
    return True


def backup_file(filepath):
    """Create a backup of the file"""
    backup_path = f"{filepath}.backup"
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"✅ Created backup: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to backup {filepath}: {e}")
        return False


def has_llm_import(content):
    """Check if file already has enhanced LLM import"""
    return "from enhanced_llm_config import" in content or "get_enhanced_llm_client" in content


def add_llm_import(content):
    """Add enhanced LLM import to file"""
    if has_llm_import(content):
        logger.info("   ✓ LLM import already present")
        return content
    
    # Find import section (after docstring, before class definition)
    lines = content.split('\n')
    import_index = -1
    
    # Find last import or first logger definition
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_index = i
        elif 'logger = logging.getLogger' in line:
            import_index = i
            break
    
    if import_index == -1:
        logger.warning("   ⚠ Could not find import section, adding at top")
        import_index = 10  # After docstring
    
    # Insert LLM import before logger
    lines.insert(import_index, LLM_IMPORT)
    logger.info("   ✓ Added LLM import")
    return '\n'.join(lines)


def add_llm_initialization(content):
    """Add LLM initialization to __init__ method"""
    if 'self.llm_client = get_enhanced_llm_client()' in content:
        logger.info("   ✓ LLM initialization already present")
        return content
    
    # Find __init__ method and suitable insertion point
    lines = content.split('\n')
    init_start = -1
    insert_index = -1
    
    for i, line in enumerate(lines):
        if 'def __init__' in line:
            init_start = i
        elif init_start > 0 and insert_index == -1:
            # Look for logger.info at end of __init__ or first method definition after __init__
            if 'logger.info' in line and '✅' in line:
                insert_index = i + 1
                break
            elif (line.strip().startswith('def ') and i > init_start + 5):
                insert_index = i
                break
    
    if insert_index == -1:
        logger.warning("   ⚠ Could not find suitable insertion point for LLM init")
        return content
    
    # Insert initialization code with proper indentation
    indent = '        '  # Assuming standard 4-space or 8-space indentation
    init_lines = INIT_CODE.strip().split('\n')
    for line in reversed(init_lines):
        if line.strip():  # Skip empty lines
            lines.insert(insert_index, line)
    
    logger.info("   ✓ Added LLM initialization")
    return '\n'.join(lines)


def update_handler(filepath):
    """Update a single handler file"""
    logger.info(f"\n📝 Processing: {filepath}")
    
    if not check_file_exists(filepath):
        return False
    
    # Read file content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"❌ Failed to read {filepath}: {e}")
        return False
    
    # Backup original file
    if not backup_file(filepath):
        return False
    
    # Apply updates
    original_content = content
    content = add_llm_import(content)
    content = add_llm_initialization(content)
    
    # Check if any changes were made
    if content == original_content:
        logger.info("   ℹ No changes needed")
        return True
    
    # Write updated content
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ Successfully updated: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write {filepath}: {e}")
        # Restore backup
        try:
            with open(f"{filepath}.backup", 'r', encoding='utf-8') as f:
                backup_content = f.read()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            logger.info(f"   ✓ Restored from backup")
        except:
            logger.error(f"   ❌ Failed to restore backup!")
        return False


def main():
    """Main integration process"""
    logger.info("=" * 80)
    logger.info("🚀 Starting Enhanced LLM Integration for All Handlers")
    logger.info("=" * 80)
    
    success_count = 0
    failed_count = 0
    
    for handler in HANDLERS:
        if update_handler(handler):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 INTEGRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Successfully updated: {success_count} handlers")
    logger.info(f"❌ Failed: {failed_count} handlers")
    logger.info(f"📁 Total processed: {len(HANDLERS)} handlers")
    logger.info("=" * 80)
    
    if failed_count == 0:
        logger.info("\n🎉 All handlers successfully integrated with Enhanced LLM!")
        logger.info("\nNext steps:")
        logger.info("1. Start the Google Cloud LLM API server: ssh into VM and run llm_api_server.py")
        logger.info("2. Set GOOGLE_CLOUD_LLM_ENDPOINT in environment or llm_config.py")
        logger.info("3. Test each handler with sample queries")
        logger.info("4. Monitor logs for LLM performance and errors")
        return 0
    else:
        logger.warning(f"\n⚠️ Integration completed with {failed_count} failures")
        logger.warning("Please review the errors above and update handlers manually")
        return 1


if __name__ == "__main__":
    sys.exit(main())
