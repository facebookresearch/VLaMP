# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from setuptools import setup, find_packages
import os

PATH_ROOT = os.path.dirname(__file__)
with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = {}  # type: ignore
with open("src/vlamp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


install_requires = [
    "matplotlib>=3.5.2",
    "networkx==2.8.1",
    "numpy==1.22.4",
    "black==22.3.0",
    "img2pdf",
    "jupyter>=1.0.0",
    "matplotlib==3.5.2",
    "networkx==2.8.1",
    "notebook==6.4.12",
    "numpy>=1.22.4",
    "tqdm",
    "pandas",
]

setup(
    name="task_structure_learning",
    version=VERSION["VERSION"],
    description="Internal repo for task structure learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/facebookresearch/VLaMP/tree/main",
    project_urls={
        "Documentation": "https://task-structure-learning.readthedocs.io",
        "Source Code": "https://https://github.com/facebookresearch/VLaMP/tree/main",
    },
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "examples",
            "wandb",
        ],
    ),
    package_dir={"": "src"},
    install_requires=install_requires,
    keywords=[
        "AI",
        "ML",
        "Machine Learning",
        "Deep Learning",
    ],
    entry_points={
        "console_scripts": [
            # TODO: Add the analysis scripts
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires=">=3.7",
)
