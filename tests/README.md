# AI Istanbul Chatbot - Test Suite

This directory contains comprehensive tests for the AI Istanbul chatbot backend.

## Test Structure

- `conftest.py` - Test configuration and fixtures
- `test_api_endpoints.py` - API endpoint testing
- `test_ai_multilingual.py` - AI and multilingual functionality tests
- `test_ai_coverage.py` - AI service coverage tests
- `test_gdpr_compliance.py` - GDPR compliance tests
- `test_performance.py` - Performance and load tests
- `test_integration.py` - End-to-end integration tests
- `test_security.py` - Security and rate limiting tests

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests with coverage
pytest --cov=backend --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest tests/test_ai_multilingual.py -v
pytest tests/test_api_endpoints.py -v

# Run with coverage threshold (70% minimum)
pytest --cov=backend --cov-fail-under=70
```

## Coverage Goals

- **Overall Backend Coverage**: >70%
- **AI Services Coverage**: >80%
- **API Endpoints Coverage**: >90%
- **Critical Path Coverage**: 100%
