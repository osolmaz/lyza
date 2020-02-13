# LYZA

This project grew out of dissatisfaction with not being able to control what is
going on under the hood with most finite elements packages. I was inspired by
[FEnICS](https://fenicsproject.org/) when I created the library. I used it to
solve nonlinear and coupled problems I encountered in courses I took.

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

## Examples

To run an example, simply go into the corresponding directory and execute the
Python file inside:

```
cd examples/hyperelasticity
python hyperelasticity.py
```

The blog posts given below contain the formulations for some of the examples.

- `examples/poisson`: [Linear Finite Elements
](https://solmaz.io/2017/11/14/linear-finite-elements/)
- `examples/linear_elasticity`: [Vectorial Finite Elements
](https://solmaz.io/2017/11/21/vectorial-finite-elements/)
- `examples/cahn-hilliard`: [Nonlinear Finite Elements
](https://solmaz.io/2017/12/22/nonlinear-finite-elements/)
- `examples/hyperelasticity`: [Variational Formulation of Elasticity
](https://solmaz.io/2018/04/01/variational-formulation-elasticity/)
- `examples/nonlinear_poisson`: [Nonlinear Finite Elements
](https://solmaz.io/2017/12/22/nonlinear-finite-elements/)
- `examples/reaction_advection_diffusion`: [Time-Dependent Finite Elements
](https://solmaz.io/2017/12/07/time-dependent-finite-elements/)

The examples are specified and solved in very simple domains, but it is possible
to read in or generate arbitrary meshes.
