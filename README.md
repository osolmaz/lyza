# LYZA Prototype

## Installation

For basic usage:

    cd lyza_prototype
    sudo python setup.py install


For development:

    cd lyza_prototype/
    unset PETSC_DIR
    unset PETSC_ARCH
    pipenv install
    pipenv shell
    sudo pip install --editable .
