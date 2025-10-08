# AI Istanbul Load Testing Suite

This comprehensive load testing suite evaluates the performance of the AI Istanbul application under various load conditions.

## 🎯 Testing Scope

### Backend API Endpoints
- **Chat System**: `/ai`, `/ai/stream`, `/ai/chat`
- **Restaurant Search**: `/restaurants/search`, `/restaurants/enhanced-search`
- **Places API**: `/places/`, various place endpoints
- **Blog System**: `/blog/`, `/blog/{id}`, blog CRUD operations
- **Route Planning**: `/api/routes/generate`, route optimization
- **Location Services**: `/location/`, live location tracking
- **Cache Monitoring**: `/api/cache/analytics`, cache performance
- **Authentication**: `/auth/login`, admin endpoints

### Frontend Performance
- **React App Loading**: Initial bundle size and load time
- **API Integration**: Frontend-backend communication
- **Real-time Features**: Streaming responses, live updates
- **Mobile Responsiveness**: Touch interactions, viewport handling

### Infrastructure Components
- **Database Performance**: SQLAlchemy queries, connection pooling
- **Redis Caching**: Cache hit/miss ratios, TTL optimization
- **File Uploads**: Blog images, static assets
- **External APIs**: Google Places, OpenAI integration

## 🧪 Test Categories

### 1. Stress Testing (`stress_test.py`)
- High concurrent user simulation
- Memory and CPU usage monitoring
- Database connection limits
- API rate limiting validation

### 2. Load Testing (`api_load_test.py`)
- Realistic user load simulation
- Response time measurement
- Throughput analysis
- Error rate monitoring

### 3. Endurance Testing (`endurance_test.py`)
- Long-running performance validation
- Memory leak detection
- Resource cleanup verification
- Database connection stability

### 4. Integration Testing (`integration_test.py`)
- End-to-end workflow testing
- Multi-service coordination
- Data consistency validation
- Error propagation testing

### 5. Frontend Performance (`frontend_performance.py`)
- Bundle size analysis
- Loading time measurement
- API response handling
- User interaction simulation

## 🚀 Quick Start

### Option 1: Automated Setup & Execution
```bash
# Setup environment (one-time)
cd load-testing
python setup.py

# Run all tests
python run_tests.py

# Run specific tests
python run_tests.py --tests load stress
python run_tests.py --env production
python run_tests.py --list  # Show available tests
```

### Option 2: Using Make Commands
```bash
cd load-testing
make setup          # Setup environment
make test           # Run all tests
make test-quick     # Run quick test suite
make test-prod      # Test production environment
make report         # Generate HTML report
make clean          # Clean old results
```

### Option 3: Individual Test Execution
```bash
# Install dependencies first
pip install -r requirements.txt

# Run individual tests
python api_load_test.py --env local
python stress_test.py --env local
python endurance_test.py --env local
python integration_test.py --env local
python frontend_performance.py --env local

# Generate performance report
python generate_report.py
```

## 📊 Metrics Collected

- **Response Times**: p50, p95, p99 percentiles
- **Throughput**: Requests per second
- **Error Rates**: 4xx, 5xx error percentages
- **Resource Usage**: CPU, Memory, Disk I/O
- **Database Performance**: Query times, connection count
- **Cache Performance**: Hit ratios, eviction rates
- **Network Metrics**: Bandwidth usage, latency

## 🔧 Configuration

Edit `config.py` to customize:
- Target URLs (local/production)
- Test parameters (users, duration)
- Monitoring thresholds
- Report formatting options

## 📈 Performance Targets

### API Response Times (95th percentile)
- **Chat Endpoints**: < 2000ms
- **Search Endpoints**: < 1000ms
- **CRUD Operations**: < 500ms
- **Static Content**: < 100ms

### Throughput Targets
- **Concurrent Users**: 100+ simultaneous
- **Requests/Second**: 50+ sustained
- **Database Queries**: 200+ QPS

### Resource Limits
- **Memory Usage**: < 512MB per instance
- **CPU Usage**: < 80% sustained
- **Error Rate**: < 1% under normal load

## 🏗️ Architecture Support

This test suite is designed for the AI Istanbul architecture:
- **Frontend**: React 19 + Vite + Tailwind CSS
- **Backend**: FastAPI + SQLAlchemy + Redis
- **Database**: PostgreSQL/SQLite
- **External APIs**: OpenAI GPT-3.5, Google Places
- **Deployment**: Vercel (frontend) + Railway/Render (backend)
