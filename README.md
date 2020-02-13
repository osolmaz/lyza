# LYZA



## Installation

For basic usage:

    cd lyza
    sudo python setup.py install


For development:

    cd lyza/
    unset PETSC_DIR
    unset PETSC_ARCH
    pipenv install
    pipenv shell
    sudo pip install --editable .
