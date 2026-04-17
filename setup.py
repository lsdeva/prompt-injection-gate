from setuptools import setup, find_packages

setup(
    name="prompt-injection-gate",
    version="0.1.0",
    description="Multi-stage prompt injection detection pipeline",
    author="deva@soa.team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.36",
        "fastapi>=0.109",
        "uvicorn>=0.27",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.4",
        "pandas>=2.1",
        "pyarrow>=14.0",
        "tqdm>=4.66",
        "requests>=2.31",
    ],
    extras_require={
        "claude": ["anthropic>=0.40"],
        "train": ["datasets>=2.16", "accelerate>=0.25", "datasketch>=1.6"],
        "test": ["pytest>=8.0", "httpx>=0.26", "pytest-asyncio>=0.23"],
    },
)
