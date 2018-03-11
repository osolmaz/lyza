from setuptools import setup, find_packages
from os import path
from setuptools.extension import Extension


here = path.abspath(path.dirname(__file__))


setup(
    name = 'lyza_prototype',

    version = '0.0',

    description = '',

    author = 'H. Onur Solmaz',

    author_email = 'onursolmaz@gmail.com',

    packages = find_packages(exclude=['contrib', 'docs', 'tests']),

    extras_require = {
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    setup_requires = [
        'numpy',
        'sympy',
        'scipy',
        'petsc',
        'petsc4py',
        'progressbar2',
    ],

    entry_points = {
        'console_scripts': [
            # 'fzfopen_search=fzfopen.main:fzfopen_search',
        ],
    },
)



