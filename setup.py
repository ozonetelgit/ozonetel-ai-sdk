#!/usr/bin/env python
__author__ = 'Biswajit Satapathy'
__this__='ozonetel-ai'

import os
from setuptools import setup, find_packages
from ozoneai import __version__ as version

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

install_requires=[
    "requests",
    "numpy"
]

package_name = "ozonetel-ai"
setup(
    name = package_name,
    version = version,
    author = "Biswajit Satapathy",
    author_email = "biswajit@ozonetel.com",
    description = ("The Ozonetel AI project is designed to provide a user-friendly interface for software development using Ozonetel's in-house AI libraries, models, and software solutions. It offers seamless integration with Ozonetel's advanced AI capabilities, allowing developers to harness the power of AI to enhance their applications."),
    license = ("MIT License"),
    keywords = "Machine Learning, Artificial Intellegence, Neural Network, Indexing, Searching",
    url = "",
    packages = ["ozoneai"],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_version = ">=3.8",
    install_requires = install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
