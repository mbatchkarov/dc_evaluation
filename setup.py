# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='dc_evaluation',
    version='1.0',
    packages=['eval'],
    author=['Miroslav Batchkarov'],
    author_email=['M.Batchkarov@sussex.ac.uk'],
    install_requires=['pandas', 'numpy', 'scipy',
                      'scikit-learn', 'joblib', 'configobj',
                      'networkx'])

