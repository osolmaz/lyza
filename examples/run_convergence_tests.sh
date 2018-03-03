echo 'Nonlinear Poisson'
cd nonlinear_poisson && python convergence_test.py && cd ..
echo 'Poisson'
cd poisson && python convergence_test.py && cd ..
echo 'Linear Elasticity'
cd linear_elasticity && python convergence_test.py && cd ..
echo 'RAD'
cd reaction_advection_diffusion && python convergence_test.py && cd ..

