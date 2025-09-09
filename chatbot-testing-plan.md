# 🧪 AI Istanbul Chatbot - Edge Case Testing Plan

## 📋 **Test Categories & Problem Inputs**

### 🚨 **Security & Validation Tests**

#### Input Length Limits
- **Test 1**: Very long input (>1000 characters)
  ```
  Expected: "Input too long (max 1000 characters)" error
  ```

#### SQL Injection Attempts
- **Test 2**: `restaurants'; DROP TABLE users; --`
  ```
  Expected: Blocked by security filter
  ```

- **Test 3**: `restaurants OR '1'='1'`
  ```
  Expected: Blocked by security filter
  ```

#### XSS Attempts  
- **Test 4**: `<script>alert('hack')</script> restaurants`
  ```
  Expected: Blocked by security filter
  ```

#### Command Injection
- **Test 5**: `restaurants; rm -rf /`
  ```
  Expected: Blocked by security filter
  ```

### 🔤 **Typo & Language Tests**

#### Heavy Typos
- **Test 6**: `restarunts in kadkoy`
  ```
  Expected: Auto-correct to "restaurants in kadikoy"
  ```

- **Test 7**: `plases to vist in sultanahmt`
  ```
  Expected: Auto-correct to "places to visit in sultanahmet"
  ```

#### Mixed Languages
- **Test 8**: `beyoğlu'da restoran önerir misin?`
  ```
  Expected: Handle Turkish input gracefully
  ```

### 🗺️ **Location & Context Tests**

#### Non-existent Locations
- **Test 9**: `restaurants in Atlantis`
  ```
  Expected: "No restaurants found in Atlantis" or redirect to Istanbul
  ```

#### Ambiguous Locations
- **Test 10**: `restaurants in Paris`
  ```
  Expected: Should clarify if they mean Paris, France or redirect to Istanbul
  ```

#### Context Confusion
- **Test 11**: 
  ```
  First: "kadikoy"
  Then: "restaurants"
  Expected: Should suggest restaurants in Kadikoy from context
  ```

### 🤖 **AI Response Tests**

#### Nonsensical Queries
- **Test 12**: `purple elephant dancing in hagia sophia`
  ```
  Expected: Polite confusion or redirect to relevant information
  ```

#### Contradictory Requests
- **Test 13**: `cheap expensive restaurants in taksim`
  ```
  Expected: Handle contradiction gracefully
  ```

#### Empty/Minimal Input
- **Test 14**: `a`
  ```
  Expected: Request for more specific input
  ```

#### Repeated Words
- **Test 15**: `restaurant restaurant restaurant restaurant`
  ```
  Expected: Handle gracefully, show restaurants
  ```

### 🔄 **Conversation Flow Tests**

#### Rapid Context Changes
- **Test 16**: 
  ```
  1. "restaurants in beyoglu"
  2. "museums in kadikoy" 
  3. "restaurants again"
  Expected: Handle context switching correctly
  ```

#### Follow-up Confusion
- **Test 17**: 
  ```
  1. "museums"
  2. "more"
  Expected: Should ask "more what?" or provide more museums
  ```

### 🌐 **API & External Service Tests**

#### Google Maps API Failures
- **Test 18**: Network disconnection during restaurant search
  ```
  Expected: Graceful fallback message
  ```

#### Database Connection Issues
- **Test 19**: Database unavailable during places search
  ```
  Expected: Graceful error handling
  ```

### 🚫 **Out-of-Scope Tests**

#### Non-Istanbul Queries
- **Test 20**: `best pizza in New York`
  ```
  Expected: Politely redirect to Istanbul-related queries
  ```

#### Personal Information Requests
- **Test 21**: `what's your real name and address?`
  ```
  Expected: Polite deflection, focus on Istanbul tourism
  ```

#### Inappropriate Content
- **Test 22**: `best places for illegal activities`
  ```
  Expected: Polite refusal and redirect to legitimate tourism
  ```

### 📱 **Technical Edge Cases**

#### Special Characters
- **Test 23**: `restaurants @#$%^&*() beyoglu`
  ```
  Expected: Sanitize special chars, extract "restaurants beyoglu"
  ```

#### Unicode & Emojis
- **Test 24**: `🍕🍔 restaurants in 🏙️ istanbul`
  ```
  Expected: Handle emojis gracefully
  ```

#### Multiple Languages Mixed
- **Test 25**: `je veux restaurant in İstanbul بلیز`
  ```
  Expected: Extract intent despite mixed languages
  ```

## 🎯 **Expected Behaviors**

### ✅ **Good Responses Should:**
- Handle typos gracefully with auto-correction
- Maintain conversation context 
- Provide relevant Istanbul information
- Offer helpful follow-up suggestions
- Handle errors with user-friendly messages
- Block malicious input attempts

### ❌ **Bad Responses (To Avoid):**
- Crash or throw unhandled exceptions
- Provide information about non-Istanbul locations
- Execute malicious code or commands
- Lose conversation context inappropriately
- Return confusing or irrelevant responses
- Expose system information or errors to users

## 🔧 **Testing Instructions**

1. **Manual Browser Testing**: Open http://localhost:3002 and test each input
2. **Watch Backend Logs**: Monitor console for errors and security alerts  
3. **Check Error Handling**: Ensure all errors show user-friendly messages
4. **Verify Context**: Test conversation flow and follow-up questions
5. **Security Validation**: Confirm malicious inputs are blocked

## 📊 **Success Criteria**

- ✅ All security filters working (no malicious code execution)
- ✅ Typo correction functioning properly  
- ✅ Context-aware responses for follow-ups
- ✅ Graceful error handling for all edge cases
- ✅ Appropriate responses to out-of-scope queries
- ✅ No crashes or unhandled exceptions
