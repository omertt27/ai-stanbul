#!/usr/bin/env python3
"""
Istanbul Tourism Data Pipeline Setup Script
Initializes the environment and creates necessary directories
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/validated',
        'data/training',
        'data/training/qa_format',
        'data/training/instruction_format',
        'data/training/chatml_format',
        'data/training/conversation_format',
        'logs',
        'models',
        'models/checkpoints',
        'evaluation',
        'evaluation/results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_env_file():
    """Create sample .env file"""
    env_content = """# Istanbul Tourism Data Pipeline Environment Variables

# Google Places API (optional - get from Google Cloud Console)
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here

# Data collection settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_DELAY_SECONDS=1.0
USER_AGENT=Istanbul-Tourism-Assistant/1.0

# Database settings (if using)
DATABASE_URL=sqlite:///istanbul_tourism.db

# Redis settings (for caching)
REDIS_URL=redis://localhost:6379

# Logging level
LOG_LEVEL=INFO

# Model training settings
MODEL_NAME=istanbul-tourism-assistant
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=16
LEARNING_RATE=5e-5

# Production settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        logger.info("Created .env file with sample configuration")
    else:
        logger.info(".env file already exists")

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path('requirements.txt')
    if requirements_file.exists():
        logger.info("Installing Python requirements...")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], check=True)
            logger.info("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False
    else:
        logger.warning("requirements.txt not found")
    
    return True

def check_system_requirements():
    """Check system requirements"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8+ is required")
        return False
    
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    if free_gb < 5:
        logger.warning(f"Low disk space: {free_gb:.1f}GB available. Consider freeing up space.")
    else:
        logger.info(f"Disk space: {free_gb:.1f}GB available")
    
    return True

def create_sample_config():
    """Create sample configuration files"""
    
    # Create sample data source configuration
    sample_config = """# Sample Data Collection Configuration
# Copy this to config_local.py and customize

COLLECTION_CONFIG = {
    'max_concurrent_requests': 5,
    'request_delay': 1.0,
    'retry_attempts': 3,
    'timeout_seconds': 30,
    
    'data_sources': {
        'istanbul_tourism_official': {
            'enabled': True,
            'priority': 1,
            'rate_limit': 30
        },
        'google_places': {
            'enabled': False,  # Requires API key
            'priority': 2,
            'rate_limit': 100
        }
    },
    
    'quality_thresholds': {
        'min_text_length': 50,
        'max_text_length': 5000,
        'min_relevance_score': 0.3
    }
}
"""
    
    config_path = Path('config_sample.py')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(sample_config)
        logger.info("Created sample configuration file")

def main():
    """Main setup function"""
    logger.info("=" * 50)
    logger.info("Istanbul Tourism Data Pipeline Setup")
    logger.info("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements not met")
        return 1
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_sample_config()
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        return 1
    
    logger.info("=" * 50)
    logger.info("Setup completed successfully!")
    logger.info("=" * 50)
    
    print("\nNext steps:")
    print("1. Edit .env file with your API keys (optional)")
    print("2. Run data collection: python pipeline_runner.py --collection-only")
    print("3. Run full pipeline: python pipeline_runner.py")
    print("4. Check logs/ directory for execution logs")
    print("\nFor help: python pipeline_runner.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
