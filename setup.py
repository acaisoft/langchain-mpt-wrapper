from setuptools import setup, find_packages

VERSION = '0.0.5'

setup(
    name='mpt_wrapper',
    version=VERSION,
    packages=find_packages(include=['mpt_wrapper',]),
    install_requires=[
        'langchain==0.0.173',
        'torch==2.0.1 ',
        'transformers==4.29.2',
    ]
)