from setuptools import setup, find_packages
from os import path
from setuptools.extension import Extension


here = path.abspath(path.dirname(__file__))


setup(
    name = "pylyza",

    version = "0.0",

    description = "",

    author = "H. Onur Solmaz",

    author_email = "onursolmaz@gmail.com",

    packages = find_packages(exclude=["contrib", "docs", "tests"]),

    extras_require = {
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },

    entry_points = {
        "console_scripts": [
            # "fzfopen_search=fzfopen.main:fzfopen_search",
            # "fzfopen=fzfopen.main:fzfopen",
        ],
    },
)



