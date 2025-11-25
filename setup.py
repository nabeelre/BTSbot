"""
Setup script for BTSbot package.
This file is provided for backward compatibility with older installation methods.
For new installations, use: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="btsbot",
    version="2.0.0",
    author="Nabeel Rehemtulla",
    author_email="nabeelr@u.northwestern.edu",
    description="Multi-modal convolutional neural network for automating supernova identification and follow-up in the Zwicky Transient Facility (ZTF) Bright Transient Survey (BTS)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/nabeelre/BTSbot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.2.2",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn==1.5.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.8.0",
        "matplotlib==3.7.1",
        "tqdm==4.66.4",
        "requests==2.32.3",
        "astropy==6.0.1",
        "pymongo==4.7.0",
        "bson>=0.5.10",
        "penquins==2.4.0",
        "timm>=0.9.0",
        "umap-learn>=0.5.0",
        "wandb>=0.15.0",
        "datasets>=2.14.0",
        "prettytable==3.10.0",
        "huggingface-hub>=0.16.0",
    ],
    extras_require={
        "gpu": ["nvtx>=0.2.0"],
    },
)

