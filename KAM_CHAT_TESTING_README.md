# KAM AI Chat - Comprehensive Testing Suite

This testing suite includes **40 test cases** covering all major features of the KAM AI chat system for Istanbul restaurants and attractions.

---

## 📦 What's Included

### 1. **Test Configuration** (`test_kam_chat_comprehensive.json`)
- 40 comprehensive test cases
- 20 restaurant tests
- 20 places & attractions tests
- Expected features and criteria for each test

### 2. **Automated Test Runner** (`run_kam_chat_tests.py`)
- Python script to execute all tests automatically
- Analyzes responses against expected features
- Generates detailed test reports
- Color-coded terminal output

### 3. **Manual Test Checklist** (`KAM_CHAT_TEST_MANUAL_CHECKLIST.md`)
- Printable/interactive checklist format
- Detailed evaluation criteria
- Scoring guidelines
- Notes template

---

## 🚀 Quick Start

### Option A: Automated Testing

1. **Start the backend server:**
   ```bash
   cd /Users/omer/Desktop/ai-stanbul
   python app.py
   # OR
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. **Run the automated test suite:**
   ```bash
   python run_kam_chat_tests.py
   ```

3. **View results:**
   - Terminal will show real-time test execution
   - Results saved to `test_results_[timestamp].json`

### Option B: Manual Testing

1. **Open the chat interface** in your browser
2. **Follow the checklist** in `KAM_CHAT_TEST_MANUAL_CHECKLIST.md`
3. **Test each of the 40 cases** systematically
4. **Record observations** and scores

---

## 📋 Test Categories

### 🍽️ Restaurant Tests (20 tests)

1. **Location-Specific** (4 tests)
   - Beyoğlu, Sultanahmet, Kadıköy, Taksim

2. **Cuisine Filtering** (3 tests)
   - Turkish, Seafood, Street Food

3. **Dietary Restrictions** (4 tests)
   - Vegetarian, Vegan, Halal, Gluten-free

4. **Price Level** (2 tests)
   - Budget-friendly, Fine dining

5. **Typo Correction** (2 tests)
   - Various spelling errors

6. **Operating Hours** (1 test)
   - "Open now" queries

7. **Natural Language** (2 tests)
   - Conversational queries

8. **Combined Filters** (2 tests)
   - Multiple criteria

### 🏛️ Places & Attractions Tests (20 tests)

1. **General Attractions** (1 test)
   - Overview of top sites

2. **District-Specific** (3 tests)
   - Sultanahmet, Beyoğlu, Kadıköy

3. **Category Filtering** (4 tests)
   - Museums, Monuments, Parks, Religious Sites

4. **Weather-Appropriate** (2 tests)
   - Indoor/Outdoor based on weather

5. **Special Interest** (2 tests)
   - Family-friendly, Romantic

6. **Budget-Friendly** (2 tests)
   - Free attractions, Affordable activities

7. **Typo Correction** (2 tests)
   - Spelling error handling

8. **Natural Language** (2 tests)
   - Conversational understanding

9. **Combined Filters** (2 tests)
   - Multi-criteria searches

---

## 🎯 Test Coverage

### Features Tested:

✅ **Location Filtering**
- District-based search (Beyoğlu, Sultanahmet, etc.)
- European vs Asian side recognition
- Neighborhood accuracy

✅ **Cuisine & Category Filtering**
- Turkish, Seafood, Vegetarian cuisines
- Museums, Parks, Monuments categories
- Religious sites identification

✅ **Dietary Restrictions**
- Vegetarian, Vegan options
- Halal, Kosher requirements
- Gluten-free, Allergy-friendly

✅ **Price Level Indicators**
- Budget-friendly (💰)
- Moderate (💰💰)
- Expensive (💰💰💰)
- Fine dining (💰💰💰💰)

✅ **Operating Hours**
- "Open now" status
- Hours of operation display
- Real-time availability

✅ **Smart Typo Correction**
- Common misspellings
- Location name variations
- Fuzzy matching

✅ **Context-Aware Processing**
- Natural language understanding
- Intent recognition
- Follow-up capability

✅ **Weather-Appropriate Suggestions**
- Indoor vs outdoor filtering
- Seasonal recommendations
- Weather-based activities

✅ **Special Interest Tags**
- Family-friendly
- Romantic spots
- Tourist vs local experiences

✅ **Budget Filtering**
- Free attractions
- Low-cost activities
- Value-based filtering

---

## 📊 Evaluation Metrics

Each test is scored on:

1. **Response Accuracy** (0-10)
   - Relevance to query
   - Correctness of information

2. **Feature Detection** (0-10)
   - Expected features present
   - Filtering working correctly

3. **Data Quality** (0-10)
   - Complete information
   - Accurate details

4. **Performance** (0-10)
   - Response time
   - No errors

5. **User Experience** (0-10)
   - Clarity and helpfulness
   - Formatting quality

**Total Score per Test:** 0-100

---

## 📈 Success Criteria

### Minimum Requirements:
- ✅ 95% tests pass (38/40)
- ✅ Average score ≥ 75/100
- ✅ No API errors
- ✅ Response time < 3 seconds
- ✅ 100% typo correction accuracy
- ✅ 100% location filtering accuracy

### Excellent Performance:
- 🌟 100% tests pass (40/40)
- 🌟 Average score ≥ 85/100
- 🌟 Response time < 2 seconds
- 🌟 All special features working
- 🌟 Excellent natural language understanding

---

## 🔧 Requirements

### For Automated Testing:
```bash
pip install requests
```

### Backend Server:
- FastAPI server running on `http://localhost:8000`
- All API endpoints operational
- Database connected

### Frontend:
- React app running on `http://localhost:3000` (or your port)
- Chat interface accessible

---

## 📝 Test Examples

### Restaurant Test Example:
```
Input: "Give me restaurant advice in Beyoğlu"
Expected:
- 4 restaurant recommendations
- All in Beyoğlu district
- Ratings and prices shown
- Addresses included
- Operating hours displayed
```

### Attraction Test Example:
```
Input: "Free family friendly museums in Beyoğlu"
Expected:
- Museum-type attractions
- In Beyoğlu district
- Free or low-cost entry
- Family-appropriate
- Age recommendations
```

---

## 🐛 Reporting Issues

When you find issues during testing:

1. **Note the test ID** (1-40)
2. **Record the exact input** used
3. **Capture the response** (screenshot or text)
4. **Describe expected vs actual** behavior
5. **Score the test** (0-100)
6. **Add to bug tracker**

---

## 📊 Results Format

Automated tests generate JSON results:

```json
{
  "test_id": 1,
  "input": "Give me restaurant advice in Beyoğlu",
  "score": 85.5,
  "response_time": 1.2,
  "features_detected": ["location_filtering", "4_results"],
  "features_missing": [],
  "issues": []
}
```

---

## 🔄 Continuous Testing

### Recommended Testing Schedule:
- **After major features:** Run full suite
- **Before deployment:** Run full suite
- **Weekly:** Run spot checks (10 random tests)
- **Daily:** Monitor response times

---

## 📧 Support

For questions or issues with the test suite:
- Review test configurations
- Check API connectivity
- Verify backend is running
- Ensure all dependencies installed

---

## 🎉 Getting Started Now

**Quickest way to test:**

1. Start backend: `python app.py` (or `uvicorn app:app`)
2. Run tests: `python run_kam_chat_tests.py`
3. Review results in terminal
4. Check detailed JSON report

**Estimated Time:** 5-10 minutes for automated run

---

*Happy Testing! 🚀*
