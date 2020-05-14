from setuptools import setup, find_packages
from os import path
from setuptools.extension import Extension

with open("requirements.txt") as f:
    required = f.read().splitlines()

here = path.abspath(path.dirname(__file__))


setup(
    name="lyza",
    version="0.0",
    description="",
    author="Onur Solmaz",
    author_email="onursolmaz@gmail.com",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    extras_require={"dev": ["check-manifest"], "test": ["coverage"],},
    install_requires=required,
    entry_points={
        "console_scripts": [
            # 'fzfopen_search=fzfopen.main:fzfopen_search',
        ],
    },
)
