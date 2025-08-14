from setuptools import setup, find_packages

setup(
    name="geo-rag-system",
    version="0.1.0",
    description="Geographic Information RAG System with Spatial Intelligence",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.8",
)
