"""
Installation script
"""
from setuptools import setup, find_packages

setup(
    version='0.0.1',
    name='gradientflow',
    packages=find_packages(exclude=["build"])
)
