from setuptools import setup, find_packages

setup(
    name="sekai-memory",
    version="1.0.0",
    description="Sekai's Multi-Character Memory System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "asyncpg>=0.29.0",
        "psycopg2-binary>=2.9.9",
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "httpx>=0.25.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.12.0",
        ],
        "optional": [
            "openai>=1.0.0",
            "pandas>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sekai-memory-server=app.main:main",
            "sekai-memory-seed=db.seed:main",
            "sekai-memory-load=scripts.load_json:main",
        ],
    },
)