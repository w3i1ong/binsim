from setuptools import setup
from Cython.Build import cythonize
if __name__ == '__main__':
    setup(ext_modules=cythonize('pyIDAIns.pyx'))
