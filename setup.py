#!/usr/bin/env python
__author__ = 'Biswajit Satapathy'
__this__='ozonetel-ai'

import os
from setuptools import setup, find_packages
from ozoneai import __version__ as version

__this__="ozonetel-ai"
__src__ = os.path.join(os.getcwd(),'src/')

srcs=[]
scripts=[]

install_requires=[]
with open('requirements.txt','r') as fp:
    install_requires = [d.strip() for d in fp.readlines()]
    fp.close()

package_name = "ozonetel-ai"
setup(
    name = package_name,
    version = version,
    author = "Biswajit Satapathy",
    author_email = "biswajit@ozonetel.com",
    description = ("The Ozonetel AI Library is a powerful tool developed by Ozonetel Communications Pvt Ltd, designed to empower developers with state-of-the-art Artificial Intelligence capabilities. This library provides seamless integration of advanced AI models into your projects, allowing you to leverage the latest in AI technology to enhance your applications."),
    license = ("Ozonetel"),
    keywords = "Machine Learning, Artificial Intellegence, Neural Network, Indexing, Searching",
    url = "",
    packages = ["ozoneai"],
    python_version = ">=3.8",
    install_requires = install_requires
)