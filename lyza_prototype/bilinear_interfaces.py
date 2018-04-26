import numpy as np
from lyza_prototype.element_interface import ElementInterface
from lyza_prototype.quantity import Quantity
import itertools

def delta(i,j):
    if i==j:
        return 1
    else:
        return 0


def to_voigt(matrix):
    result = np.zeros((6,1))
    voigt_index_map = [[0,3,5],[3,1,4],[5,4,2]]
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        result[voigt_index_map[i][j]] = matrix[i,j]
    return result


class PoissonMatrix(ElementInterface):

    def matrix(self):

        K = np.zeros((self.elements[1].n_dof, self.elements[0].n_dof))

        for q1, q2 in zip(self.elements[0].quad_points, self.elements[1].quad_points):

            for I,J,i in itertools.product(
                    range(self.elements[0].n_node),
                    range(self.elements[1].n_node),
                    range(self.elements[0].spatial_dimension)):

                K[I, J] += q1.B[I][i]*q2.B[J][i]*q1.det_jac*q1.weight

        return K


class MassMatrix(ElementInterface):

    def matrix(self):

        K = np.zeros((self.elements[1].n_dof, self.elements[0].n_dof))

        for q1, q2 in zip(self.elements[0].quad_points, self.elements[1].quad_points):

            for I,J in itertools.product(
                    range(self.elements[0].n_node),
                    range(self.elements[1].n_node)):

                K[I, J] += q1.N[I]*q2.N[J]*q1.det_jac*q1.weight

        return K


class LinearElasticity(ElementInterface):

    def __init__(self, C, plane_stress=False, plane_strain=False, thickness=None):

        if plane_stress and plane_strain:
            raise Exception('Can be either plane stress or plane strain')

        # # Plane stress
        # self.C = E/(1-nu*nu)*np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1.-nu)/2.]])
        # self.index_map = [[0,2],[2,1]]

        # # Plane strain
        # self.C = E/(1.+nu)/(1-2.*nu)*np.array([[1.-nu,nu,0.],[nu,1.-nu,0.],[0.,0.,(1.-2.*nu)/2.]])
        # self.index_map = [[0,2],[2,1]]


        if plane_strain:
            self.C = np.array([[C[0,0], C[0,1], C[0,3]], [C[1,0], C[1,1], C[1,3]], [C[3,0], C[3,1], C[3,3]]])
            self.index_map = [[0,2],[2,1]]
        elif plane_stress:
            self.C = np.array([
                [(C[0,0]*C[2,2]*C[4,4]*C[5,5] - C[0,0]*C[2,2]*C[4,5]*C[5,4] - C[0,0]*C[2,4]*C[4,2]*C[5,5] + C[0,0]*C[2,4]*C[4,5]*C[5,2] + C[0,0]*C[2,5]*C[4,2]*C[5,4] - C[0,0]*C[2,5]*C[4,4]*C[5,2] - C[0,2]*C[2,0]*C[4,4]*C[5,5] + C[0,2]*C[2,0]*C[4,5]*C[5,4] + C[0,2]*C[2,4]*C[4,0]*C[5,5] - C[0,2]*C[2,4]*C[4,5]*C[5,0] - C[0,2]*C[2,5]*C[4,0]*C[5,4] + C[0,2]*C[2,5]*C[4,4]*C[5,0] + C[0,4]*C[2,0]*C[4,2]*C[5,5] - C[0,4]*C[2,0]*C[4,5]*C[5,2] - C[0,4]*C[2,2]*C[4,0]*C[5,5] + C[0,4]*C[2,2]*C[4,5]*C[5,0] + C[0,4]*C[2,5]*C[4,0]*C[5,2] - C[0,4]*C[2,5]*C[4,2]*C[5,0] - C[0,5]*C[2,0]*C[4,2]*C[5,4] + C[0,5]*C[2,0]*C[4,4]*C[5,2] + C[0,5]*C[2,2]*C[4,0]*C[5,4] - C[0,5]*C[2,2]*C[4,4]*C[5,0] - C[0,5]*C[2,4]*C[4,0]*C[5,2] + C[0,5]*C[2,4]*C[4,2]*C[5,0])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2]),
                 (C[0,1]*C[2,2]*C[4,4]*C[5,5] - C[0,1]*C[2,2]*C[4,5]*C[5,4] - C[0,1]*C[2,4]*C[4,2]*C[5,5] + C[0,1]*C[2,4]*C[4,5]*C[5,2] + C[0,1]*C[2,5]*C[4,2]*C[5,4] - C[0,1]*C[2,5]*C[4,4]*C[5,2] - C[0,2]*C[2,1]*C[4,4]*C[5,5] + C[0,2]*C[2,1]*C[4,5]*C[5,4] + C[0,2]*C[2,4]*C[4,1]*C[5,5] - C[0,2]*C[2,4]*C[4,5]*C[5,1] - C[0,2]*C[2,5]*C[4,1]*C[5,4] + C[0,2]*C[2,5]*C[4,4]*C[5,1] + C[0,4]*C[2,1]*C[4,2]*C[5,5] - C[0,4]*C[2,1]*C[4,5]*C[5,2] - C[0,4]*C[2,2]*C[4,1]*C[5,5] + C[0,4]*C[2,2]*C[4,5]*C[5,1] + C[0,4]*C[2,5]*C[4,1]*C[5,2] - C[0,4]*C[2,5]*C[4,2]*C[5,1] - C[0,5]*C[2,1]*C[4,2]*C[5,4] + C[0,5]*C[2,1]*C[4,4]*C[5,2] + C[0,5]*C[2,2]*C[4,1]*C[5,4] - C[0,5]*C[2,2]*C[4,4]*C[5,1] - C[0,5]*C[2,4]*C[4,1]*C[5,2] + C[0,5]*C[2,4]*C[4,2]*C[5,1])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2]),
                 (-C[0,2]*C[2,3]*C[4,4]*C[5,5] + C[0,2]*C[2,3]*C[4,5]*C[5,4] + C[0,2]*C[2,4]*C[4,3]*C[5,5] - C[0,2]*C[2,4]*C[4,5]*C[5,3] - C[0,2]*C[2,5]*C[4,3]*C[5,4] + C[0,2]*C[2,5]*C[4,4]*C[5,3] + C[0,3]*C[2,2]*C[4,4]*C[5,5] - C[0,3]*C[2,2]*C[4,5]*C[5,4] - C[0,3]*C[2,4]*C[4,2]*C[5,5] + C[0,3]*C[2,4]*C[4,5]*C[5,2] + C[0,3]*C[2,5]*C[4,2]*C[5,4] - C[0,3]*C[2,5]*C[4,4]*C[5,2] - C[0,4]*C[2,2]*C[4,3]*C[5,5] + C[0,4]*C[2,2]*C[4,5]*C[5,3] + C[0,4]*C[2,3]*C[4,2]*C[5,5] - C[0,4]*C[2,3]*C[4,5]*C[5,2] - C[0,4]*C[2,5]*C[4,2]*C[5,3] + C[0,4]*C[2,5]*C[4,3]*C[5,2] + C[0,5]*C[2,2]*C[4,3]*C[5,4] - C[0,5]*C[2,2]*C[4,4]*C[5,3] - C[0,5]*C[2,3]*C[4,2]*C[5,4] + C[0,5]*C[2,3]*C[4,4]*C[5,2] + C[0,5]*C[2,4]*C[4,2]*C[5,3] - C[0,5]*C[2,4]*C[4,3]*C[5,2])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2])],
                [(C[1,0]*C[2,2]*C[4,4]*C[5,5] - C[1,0]*C[2,2]*C[4,5]*C[5,4] - C[1,0]*C[2,4]*C[4,2]*C[5,5] + C[1,0]*C[2,4]*C[4,5]*C[5,2] + C[1,0]*C[2,5]*C[4,2]*C[5,4] - C[1,0]*C[2,5]*C[4,4]*C[5,2] - C[1,2]*C[2,0]*C[4,4]*C[5,5] + C[1,2]*C[2,0]*C[4,5]*C[5,4] + C[1,2]*C[2,4]*C[4,0]*C[5,5] - C[1,2]*C[2,4]*C[4,5]*C[5,0] - C[1,2]*C[2,5]*C[4,0]*C[5,4] + C[1,2]*C[2,5]*C[4,4]*C[5,0] + C[1,4]*C[2,0]*C[4,2]*C[5,5] - C[1,4]*C[2,0]*C[4,5]*C[5,2] - C[1,4]*C[2,2]*C[4,0]*C[5,5] + C[1,4]*C[2,2]*C[4,5]*C[5,0] + C[1,4]*C[2,5]*C[4,0]*C[5,2] - C[1,4]*C[2,5]*C[4,2]*C[5,0] - C[1,5]*C[2,0]*C[4,2]*C[5,4] + C[1,5]*C[2,0]*C[4,4]*C[5,2] + C[1,5]*C[2,2]*C[4,0]*C[5,4] - C[1,5]*C[2,2]*C[4,4]*C[5,0] - C[1,5]*C[2,4]*C[4,0]*C[5,2] + C[1,5]*C[2,4]*C[4,2]*C[5,0])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2]),
                 (C[1,1]*C[2,2]*C[4,4]*C[5,5] - C[1,1]*C[2,2]*C[4,5]*C[5,4] - C[1,1]*C[2,4]*C[4,2]*C[5,5] + C[1,1]*C[2,4]*C[4,5]*C[5,2] + C[1,1]*C[2,5]*C[4,2]*C[5,4] - C[1,1]*C[2,5]*C[4,4]*C[5,2] - C[1,2]*C[2,1]*C[4,4]*C[5,5] + C[1,2]*C[2,1]*C[4,5]*C[5,4] + C[1,2]*C[2,4]*C[4,1]*C[5,5] - C[1,2]*C[2,4]*C[4,5]*C[5,1] - C[1,2]*C[2,5]*C[4,1]*C[5,4] + C[1,2]*C[2,5]*C[4,4]*C[5,1] + C[1,4]*C[2,1]*C[4,2]*C[5,5] - C[1,4]*C[2,1]*C[4,5]*C[5,2] - C[1,4]*C[2,2]*C[4,1]*C[5,5] + C[1,4]*C[2,2]*C[4,5]*C[5,1] + C[1,4]*C[2,5]*C[4,1]*C[5,2] - C[1,4]*C[2,5]*C[4,2]*C[5,1] - C[1,5]*C[2,1]*C[4,2]*C[5,4] + C[1,5]*C[2,1]*C[4,4]*C[5,2] + C[1,5]*C[2,2]*C[4,1]*C[5,4] - C[1,5]*C[2,2]*C[4,4]*C[5,1] - C[1,5]*C[2,4]*C[4,1]*C[5,2] + C[1,5]*C[2,4]*C[4,2]*C[5,1])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2]),
                 (-C[1,2]*C[2,3]*C[4,4]*C[5,5] + C[1,2]*C[2,3]*C[4,5]*C[5,4] + C[1,2]*C[2,4]*C[4,3]*C[5,5] - C[1,2]*C[2,4]*C[4,5]*C[5,3] - C[1,2]*C[2,5]*C[4,3]*C[5,4] + C[1,2]*C[2,5]*C[4,4]*C[5,3] + C[1,3]*C[2,2]*C[4,4]*C[5,5] - C[1,3]*C[2,2]*C[4,5]*C[5,4] - C[1,3]*C[2,4]*C[4,2]*C[5,5] + C[1,3]*C[2,4]*C[4,5]*C[5,2] + C[1,3]*C[2,5]*C[4,2]*C[5,4] - C[1,3]*C[2,5]*C[4,4]*C[5,2] - C[1,4]*C[2,2]*C[4,3]*C[5,5] + C[1,4]*C[2,2]*C[4,5]*C[5,3] + C[1,4]*C[2,3]*C[4,2]*C[5,5] - C[1,4]*C[2,3]*C[4,5]*C[5,2] - C[1,4]*C[2,5]*C[4,2]*C[5,3] + C[1,4]*C[2,5]*C[4,3]*C[5,2] + C[1,5]*C[2,2]*C[4,3]*C[5,4] - C[1,5]*C[2,2]*C[4,4]*C[5,3] - C[1,5]*C[2,3]*C[4,2]*C[5,4] + C[1,5]*C[2,3]*C[4,4]*C[5,2] + C[1,5]*C[2,4]*C[4,2]*C[5,3] - C[1,5]*C[2,4]*C[4,3]*C[5,2])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2])],
                [(-C[2,0]*C[3,2]*C[4,4]*C[5,5] + C[2,0]*C[3,2]*C[4,5]*C[5,4] + C[2,0]*C[3,4]*C[4,2]*C[5,5] - C[2,0]*C[3,4]*C[4,5]*C[5,2] - C[2,0]*C[3,5]*C[4,2]*C[5,4] + C[2,0]*C[3,5]*C[4,4]*C[5,2] + C[2,2]*C[3,0]*C[4,4]*C[5,5] - C[2,2]*C[3,0]*C[4,5]*C[5,4] - C[2,2]*C[3,4]*C[4,0]*C[5,5] + C[2,2]*C[3,4]*C[4,5]*C[5,0] + C[2,2]*C[3,5]*C[4,0]*C[5,4] - C[2,2]*C[3,5]*C[4,4]*C[5,0] - C[2,4]*C[3,0]*C[4,2]*C[5,5] + C[2,4]*C[3,0]*C[4,5]*C[5,2] + C[2,4]*C[3,2]*C[4,0]*C[5,5] - C[2,4]*C[3,2]*C[4,5]*C[5,0] - C[2,4]*C[3,5]*C[4,0]*C[5,2] + C[2,4]*C[3,5]*C[4,2]*C[5,0] + C[2,5]*C[3,0]*C[4,2]*C[5,4] - C[2,5]*C[3,0]*C[4,4]*C[5,2] - C[2,5]*C[3,2]*C[4,0]*C[5,4] + C[2,5]*C[3,2]*C[4,4]*C[5,0] + C[2,5]*C[3,4]*C[4,0]*C[5,2] - C[2,5]*C[3,4]*C[4,2]*C[5,0])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2]),
                 (-C[2,1]*C[3,2]*C[4,4]*C[5,5] + C[2,1]*C[3,2]*C[4,5]*C[5,4] + C[2,1]*C[3,4]*C[4,2]*C[5,5] - C[2,1]*C[3,4]*C[4,5]*C[5,2] - C[2,1]*C[3,5]*C[4,2]*C[5,4] + C[2,1]*C[3,5]*C[4,4]*C[5,2] + C[2,2]*C[3,1]*C[4,4]*C[5,5] - C[2,2]*C[3,1]*C[4,5]*C[5,4] - C[2,2]*C[3,4]*C[4,1]*C[5,5] + C[2,2]*C[3,4]*C[4,5]*C[5,1] + C[2,2]*C[3,5]*C[4,1]*C[5,4] - C[2,2]*C[3,5]*C[4,4]*C[5,1] - C[2,4]*C[3,1]*C[4,2]*C[5,5] + C[2,4]*C[3,1]*C[4,5]*C[5,2] + C[2,4]*C[3,2]*C[4,1]*C[5,5] - C[2,4]*C[3,2]*C[4,5]*C[5,1] - C[2,4]*C[3,5]*C[4,1]*C[5,2] + C[2,4]*C[3,5]*C[4,2]*C[5,1] + C[2,5]*C[3,1]*C[4,2]*C[5,4] - C[2,5]*C[3,1]*C[4,4]*C[5,2] - C[2,5]*C[3,2]*C[4,1]*C[5,4] + C[2,5]*C[3,2]*C[4,4]*C[5,1] + C[2,5]*C[3,4]*C[4,1]*C[5,2] - C[2,5]*C[3,4]*C[4,2]*C[5,1])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2]),
                 (C[2,2]*C[3,3]*C[4,4]*C[5,5] - C[2,2]*C[3,3]*C[4,5]*C[5,4] - C[2,2]*C[3,4]*C[4,3]*C[5,5] + C[2,2]*C[3,4]*C[4,5]*C[5,3] + C[2,2]*C[3,5]*C[4,3]*C[5,4] - C[2,2]*C[3,5]*C[4,4]*C[5,3] - C[2,3]*C[3,2]*C[4,4]*C[5,5] + C[2,3]*C[3,2]*C[4,5]*C[5,4] + C[2,3]*C[3,4]*C[4,2]*C[5,5] - C[2,3]*C[3,4]*C[4,5]*C[5,2] - C[2,3]*C[3,5]*C[4,2]*C[5,4] + C[2,3]*C[3,5]*C[4,4]*C[5,2] + C[2,4]*C[3,2]*C[4,3]*C[5,5] - C[2,4]*C[3,2]*C[4,5]*C[5,3] - C[2,4]*C[3,3]*C[4,2]*C[5,5] + C[2,4]*C[3,3]*C[4,5]*C[5,2] + C[2,4]*C[3,5]*C[4,2]*C[5,3] - C[2,4]*C[3,5]*C[4,3]*C[5,2] - C[2,5]*C[3,2]*C[4,3]*C[5,4] + C[2,5]*C[3,2]*C[4,4]*C[5,3] + C[2,5]*C[3,3]*C[4,2]*C[5,4] - C[2,5]*C[3,3]*C[4,4]*C[5,2] - C[2,5]*C[3,4]*C[4,2]*C[5,3] + C[2,5]*C[3,4]*C[4,3]*C[5,2])/(C[2,2]*C[4,4]*C[5,5] - C[2,2]*C[4,5]*C[5,4] - C[2,4]*C[4,2]*C[5,5] + C[2,4]*C[4,5]*C[5,2] + C[2,5]*C[4,2]*C[5,4] - C[2,5]*C[4,4]*C[5,2])]])
            self.index_map = [[0,2],[2,1]]
        else:
            self.C = C
            self.index_map = [[0,3,5],[3,1,4],[5,4,2]]

        self.thickness = thickness

    def init_quadrature_point_quantities(self, n_quad_point):
        self.stress = Quantity((6, 1), n_quad_point)
        self.strain = Quantity((6, 1), n_quad_point)


    def calculate_stress(self, function):
        V = function.function_space
        voigt_index_map = [[0,3,5],[3,1,4],[5,4,2]]
        if V != self.elements[0].function_space:
            raise Exception('Function spaces do not match')

        for n_q, q in enumerate(self.elements[0].quad_points):
            grad_u = np.zeros((V.function_size, V.spatial_dimension))

            for I in range(self.elements[0].n_node):
                val = function.get_node_val(self.elements[0].nodes[I].idx)

                for i, j in itertools.product(range(V.function_size), range(V.spatial_dimension)):
                    grad_u[i, j] += val[i]*q.B[I][j]

            strain = (grad_u + grad_u.T)/2.
            stress = np.zeros((V.function_size, V.spatial_dimension))

            for i, j, k, l in itertools.product(
                    range(V.function_size), range(V.spatial_dimension),
                    range(V.function_size), range(V.spatial_dimension)):
                stress[i,j] += self.C[self.index_map[i][j], self.index_map[k][l]]*strain[k,l]

            strain_voigt = to_voigt(strain)
            stress_voigt = to_voigt(stress)

            self.stress.vectors[n_q] = stress_voigt
            self.strain.vectors[n_q] = strain_voigt
            # import ipdb; ipdb.set_trace()


    def matrix(self):
        n_node_1 = len(self.elements[0].nodes)
        n_node_2 = len(self.elements[1].nodes)

        n_dof_1 = n_node_1*self.elements[0].function_size
        n_dof_2 = n_node_2*self.elements[1].function_size

        spatial_dim = self.elements[0].spatial_dimension

        K = np.zeros((n_dof_2,n_dof_1))

        for q1, q2 in zip(self.elements[0].quad_points, self.elements[1].quad_points):

            for I,J,i,j,k,l in itertools.product(
                    range(n_node_1),
                    range(n_node_2),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim),
                    range(spatial_dim)):

                alpha = I*spatial_dim + i
                beta = J*spatial_dim + j
                C_val = self.C[self.index_map[i][k], self.index_map[j][l]]
                K[alpha, beta] += q1.B[I][k]*C_val*q2.B[J][l]*q1.det_jac*q1.weight

        if self.thickness:
            K *= self.thickness

        return K

class IsotropicLinearElasticity(LinearElasticity):
    def __init__(self, lambda_, mu, plane_stress=False, plane_strain=False, thickness=None):
        C = np.array([
            [lambda_ + 2*mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2*mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])
        super().__init__(C, plane_stress=plane_stress, plane_strain=plane_strain, thickness=thickness)
