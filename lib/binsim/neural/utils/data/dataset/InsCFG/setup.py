from setuptools import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize('basic_block_chunk_proxy.pyx'))
