import os
from setuptools import setup, find_packages

setup(name='dragg-comp',
      license='MIT',
      version='0.5.0',
      author='Aisling Pigott and Jacob Kravits',
      author_email='aisling.pigott@colorado.edu',
      packages=find_packages(),
      install_requires=[
        'asyncio',
        'aioredis',
        'gym',
        'pathos',
        'datetime',
        'async-timeout',
        'dragg',
        'cvxopt',
        'kaleido'
        ],
      py_modules=['dragg_comp'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
     )