"""
Setup script for AIstanbul Backend
"""
from setuptools import setup, find_packages

setup(
    name="ai-istanbul-backend",
    version="1.0.0",
    description="AI Istanbul Travel Assistant Backend",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "python-multipart",
        "python-dotenv",
        "openai",
        "requests",
        "fuzzywuzzy",
        "python-levenshtein"
    ],
    package_data={
        "": ["*.sql", "*.db"]
    },
    include_package_data=True
)
