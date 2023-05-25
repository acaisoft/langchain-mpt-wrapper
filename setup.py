from setuptools import setup, find_packages

VERSION = '0.0.3'

setup(
    name='langchain_mpt_wrapper',
    version=VERSION,
    packages=find_packages(include=['langchain_mpt_wrapper',]),
    install_requires=[
        'langchain==0.0.173',
        'torch==2.0.1 ',
        'transformers==4.29.2',
        'sentence_transformers',
    ]
)