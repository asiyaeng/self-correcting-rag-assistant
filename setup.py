"""
Minimal setup.py so the package can be installed with `pip install -e .`
"""
from setuptools import setup, find_packages

setup(
    name="rag-assistant",
    version="1.0.0",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
)
