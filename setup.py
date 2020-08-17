from setuptools import setup, find_packages, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

from codecs import open
from os import path
import re


## Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


## For single-sourceing the package version
def read(*parts):
    with open(path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")



ext_modules = [
    Extension("tdse.matrix_c", 
        sources=["./c/matrix/py_matrix.cc","./c/matrix/matrix.cc"],
        include_dirs=["./c/matrix"] + get_numpy_include_dirs(),
        optional=True),
]


setup(
    name='tdse',
    version=find_version("tdse", "version.py"),
    description='Simple TDSE simulator',
    url='https://github.com/jam31118/tdse',
    author='sahn',
    author_email='jam31118@gmail.com',
    classifiers=[
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='physics, quantum mechanics, simulation',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['numpy','matplotlib','vis','nunit','ntype'],
    long_description=long_description,
    license = 'GPLv3'
)

