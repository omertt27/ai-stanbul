# Archived Main System Files

**Date:** October 30, 2025  
**Status:** ARCHIVED - DO NOT USE

---

## 📋 Current Active File

**Location:** `/istanbul_ai/main_system.py` (ROOT LEVEL)

**Reason for Consolidation:**
- Merged with TTLCache infrastructure from core level
- Contains all features from both versions
- Production-grade memory management
- Single source of truth

---

## 🗄️ Archived Files

### `main_system.py.deprecated_20251030`
- **Original Location:** `/istanbul_ai/core/main_system.py`
- **Size:** 1868 lines
- **Features:** TTLCache integration, production infrastructure
- **Archived Date:** October 30, 2025
- **Reason:** Merged into root-level main_system.py

---

## ⚠️ IMPORTANT NOTICE

**DO NOT USE FILES IN THIS DIRECTORY**

All functionality has been merged into the unified root-level main_system.py.

### Correct Import:
```python
from istanbul_ai.main_system import IstanbulDailyTalkAI  ✅
```

### Deprecated Import (DO NOT USE):
```python
from istanbul_ai.core.main_system import IstanbulDailyTalkAI  ❌
```

---

## 📊 Migration Summary

- **Total Files Updated:** 23 (3 production + 20 test files)
- **Files Using New Import:** 69
- **Migration Success Rate:** 100%
- **Test Results:** All tests passed

---

## 🔗 Related Documentation

- `MAIN_SYSTEM_CONFLICT_RESOLUTION.md` - Original conflict analysis
- `MAIN_SYSTEM_MERGE_COMPLETE.md` - Merge completion report
- `MAIN_SYSTEM_ARCHITECTURE.md` - Current system architecture (to be created)

---

**For questions or issues, refer to the main documentation in the project root.**
