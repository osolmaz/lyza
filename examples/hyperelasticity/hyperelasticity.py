from lyza_prototype import *
from mesh import Cantilever3D
from interfaces import *

import logging
logging.basicConfig(level=logging.INFO)

RESOLUTION = 15
LENGTH = 300.
HORIZONTAL_WIDTH = 20.
VERTICAL_WIDTH = 20.

E = 200.
NU = 0.3

LAMBDA = mechanics.lambda_from_E_nu(E, NU)
MU = mechanics.mu_from_E_nu(E, NU)

BENDING_LOAD = 2.*HORIZONTAL_WIDTH/2.
AXIAL_LOAD = 10.

spatial_dimension = 3
function_size = 3
element_degree = 1
quadrature_degree = 1

left_boundary = lambda x, t: x[0] <= 1e-12

up_left = lambda x, t: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] > VERTICAL_WIDTH - 1e-12
up_right = lambda x, t: x[0] > LENGTH-1e-12 and x[1] > HORIZONTAL_WIDTH - 1e-12 and x[2] > VERTICAL_WIDTH - 1e-12
down_left = lambda x, t: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] < 1e-12
down_right = lambda x, t: x[0] > LENGTH-1e-12 and x[1] > HORIZONTAL_WIDTH - 1e-12 and x[2] < 1e-12

# down_left = lambda x: x[0] > LENGTH-1e-12 and x[1] < 1e-12 and x[2] < VERTICAL_WIDTH - 1e-12

class Calculator(CellIterator):
    def init_quantities(self):
        # self.mesh.quantities['F'] = CellQuantity(self.mesh, (3, 3))
        self.mesh.quantities['BBAR'] = CellQuantity(self.mesh, None)
        self.mesh.quantities['FINVT'] = CellQuantity(self.mesh, (3, 3))
        self.mesh.quantities['LCG'] = CellQuantity(self.mesh, (3, 3))
        self.mesh.quantities['TAU'] = CellQuantity(self.mesh, (3, 3))

    def iterate(self, cell):

        F_arr = self.mesh.quantities['F'].get_quantity(cell)
        B_arr = self.mesh.quantities['B'].get_quantity(cell)

        self.mesh.quantities['BBAR'].reset_quantity_by_cell(cell)
        self.mesh.quantities['FINVT'].reset_quantity_by_cell(cell)
        self.mesh.quantities['LCG'].reset_quantity_by_cell(cell)
        self.mesh.quantities['TAU'].reset_quantity_by_cell(cell)

        for idx in range(len(F_arr)):
            F = F_arr[idx]
            B = B_arr[idx]
            Finvtra = np.linalg.inv(F).T

            # E = 0.5 * (F.T.dot(F) - identity)
            # S = self.lambda_*np.trace(E)*np.identity(3) + 2*self.mu*E
            b_ = F.dot(F.T)
            # tau = F.dot(S).dot(F.T)
            tau = (LAMBDA/2*(np.trace(b_)-3)-MU)*b_ + MU*(b_.dot(b_))

            Bbar = []
            for B_I in B:
                Bbar.append(Finvtra.dot(B_I))
            Bbar = np.array(Bbar)

            self.mesh.quantities['FINVT'].add_quantity_by_cell(cell, Finvtra)
            self.mesh.quantities['LCG'].add_quantity_by_cell(cell, b_)
            self.mesh.quantities['TAU'].add_quantity_by_cell(cell, tau)
            self.mesh.quantities['BBAR'].add_quantity_by_cell(cell, Bbar)

def update_function(mesh, phi):
    projector = iterators.GradientProjector(mesh, phi.function_size)
    projector.set_param(phi, 'F', spatial_dimension)
    projector.execute()

    calculator = Calculator(mesh, phi.function_size)
    calculator.init_quantities()
    calculator.execute()

if __name__ == '__main__':

    mesh = Cantilever3D(RESOLUTION, LENGTH, HORIZONTAL_WIDTH, VERTICAL_WIDTH)
    mesh.set_quadrature_degree(lambda c: quadrature_degree, spatial_dimension)

    a = HyperelasticityJacobian(mesh, function_size)
    a.set_param(LAMBDA, MU)
    b_res = HyperelasticityResidual(mesh, function_size)

    b_1 = vector_assemblers.PointLoadVector(mesh, function_size); b_1.set_param(up_left,[-AXIAL_LOAD,0.,-BENDING_LOAD])
    b_2 = vector_assemblers.PointLoadVector(mesh, function_size); b_2.set_param(up_right,[-AXIAL_LOAD,0.,-BENDING_LOAD])
    b_3 = vector_assemblers.PointLoadVector(mesh, function_size); b_3.set_param(down_left,[-AXIAL_LOAD,0.,0.])
    b_4 = vector_assemblers.PointLoadVector(mesh, function_size); b_4.set_param(down_right,[-AXIAL_LOAD,0.,0.])

    b_load = b_1+b_2+b_3+b_4

    phi0 = mesh.get_position_function(spatial_dimension)
    dirichlet_bcs = [DirichletBC(lambda x, t: [x[0],x[1],x[2]], left_boundary)]

    phi, residual = nonlinear_solve(a, b_res+b_load, dirichlet_bcs, initial=phi0, update_function=update_function)

    # Calculate displacement from position
    u = Function(mesh, function_size)
    u.vector = phi.vector - phi0.vector

    voigt_converter = iterators.VoigtConverter(mesh, function_size)
    voigt_converter.set_param('TAU', 'TAUV')
    voigt_converter.execute()

    tau = mesh.quantities['TAUV'].get_function()

    # Calculate internal forces
    f = Function(mesh, function_size)
    f.set_vector(-1*b_res.assemble())

    # Set labels
    u.set_label('u')
    f.set_label('f')
    phi.set_label('phi')
    tau.set_label('tau')

    # Output VTK file
    ofile = VTKFile('out_hyperelasticity.vtk')

    ofile.write(mesh, [u, phi, f, tau])



