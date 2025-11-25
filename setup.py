"""
Setup script for BTSbot package.
This file is provided for backward compatibility with older installation methods.
For new installations, use: pip install -e .
"""

from setuptools import setup, find_packages


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            # Skip comments, empty lines, and built-in modules
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements


setup(
    name="btsbot",
    version="2.0.0",
    author="Nabeel Rehemtulla",
    author_email="nabeelr@u.northwestern.edu",
    description="ML for automating SN identification and follow-up in ZTF",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/nabeelre/BTSbot",
    packages=find_packages(where="btsbot"),
    package_dir={"": "btsbot"},
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
    install_requires=read_requirements(),
    extras_require={
        "gpu": ["nvtx>=0.2.0"],
    },
)
