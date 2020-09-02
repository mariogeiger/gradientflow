"""
Installation script
"""
from setuptools import setup, find_packages

setup(
    name='gradientflow',
    packages=find_packages(exclude=["build"])
)
