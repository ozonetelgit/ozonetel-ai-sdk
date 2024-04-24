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
    description = ("The Ozonetel AI Library is a powerful tool developed by Ozonetel Communications Pvt Ltd, designed to empower developers with state-of-the-art Artificial Intelligence capabilities. This library provides seamless integration of advanced AI models into your projects, allowing you to leverage the latest in AI technology to enhance your applications."),
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
