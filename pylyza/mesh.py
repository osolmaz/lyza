import numpy as np
# import scipy as sp
# from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from math import sqrt, cos, sin, pi
import copy
# from partition_system import partition_system, partition_matrix
import itertools
import progressbar

import logging
import operator



PHYSICAL_DIM = 2


# print(quadrature_weights_2d)
# print(quadrature_points_2d)

class Node():
    def __init__(self, coor, idx, label=None):
        self.coor = coor
        self.label = label
        self.idx = idx

        self.dofmap = []
        for i in range(PHYSICAL_DIM):
            self.dofmap.append(idx*PHYSICAL_DIM+i)




class Mesh:

    def __init__(self):
        self.nodes = []
        self.elems = []
        self.edges = []

        self.node_labels = {}
        self.elem_labels = {}

        self.construct_mesh()

    def construct_mesh():
        pass

    def add_node(self, coors, label=None):
        self.nodes.append(Node(coors, len(self.nodes), label=label))

        if label:
            self.node_labels[label] = self.last_node_index()

    def last_node_index(self):
        return len(self.nodes)-1

    def last_elem_index(self):
        return len(self.elems)-1


    # def add_elem(self, nodes, param, label=None):
    #     self.elems.append(Element(nodes, param, label=label))

    #     if label:
    #         self.elem_labels[label] = self.last_elem_index()

    def add_elem(self, elem):
        self.elems.append(elem)

    def add_edge(self, elem):
        self.edges.append(elem)


    def get_elem_by_label(self, label):
        return self.elems[self.elem_labels[label]]

    # def calc_element_objects(self):
    #     bar = progressbar.ProgressBar(max_value=len(self.elems))
    #     for n, e in enumerate(self.elems):
    #         bar.update(n)
    #         e.calc_stiffness_matrix()
    #         e.calc_rhs_vector()

    def get_n_dofs(self):
        n_dofs = len(self.nodes)*PHYSICAL_DIM
        return n_dofs

    def assemble_stiffness_matrix(self):
        n_dof = self.get_n_dofs()
        K = np.zeros((n_dof,n_dof))
        # K = csr_matrix((n_dof,n_dof))

        logging.info('Calculating element matrices')
        elem_matrices = []
        bar = progressbar.ProgressBar(max_value=len(self.elems))
        for n, e in enumerate(self.elems):
            bar.update(n+1)
            elem_matrices.append(e.calc_stiffness_matrix())

        # elem_matrices = [e.calc_stiffness_matrix() for e in self.elems]

        for e, K_elem in zip(self.elems, elem_matrices):
            for i, I in enumerate(e.dofmap):
                for j, J in enumerate(e.dofmap):
                    K[I, J] += K_elem[i,j]

        return (K+K.T)/2.

    def assemble_force_rhs(self, force_function):
        n_dof = self.get_n_dofs()
        f = np.zeros((n_dof,1))

        logging.info('Calculating element force vectors')
        elem_vectors = []
        bar = progressbar.ProgressBar(max_value=len(self.elems))
        for n, e in enumerate(self.elems):
            bar.update(n+1)
            elem_vectors.append(e.calc_rhs_vector(force_function))

        # elem_vectors = [e.calc_rhs_vector(force_function) for e in self.elems]

        for e, f_elem in zip(self.elems, elem_vectors):
            for i, I in enumerate(e.dofmap):
                f[I] += f_elem[i]

        return f

    def assemble_neumann_rhs(self, neumann_bcs):
        n_dof = self.get_n_dofs()
        f = np.zeros((n_dof,1))

        for bc in neumann_bcs:
            for edge in self.edges:
                is_in = not False in [bc.position_bool(node.coor) for node in edge.nodes]
                if not is_in: continue

                f_elem = edge.calc_rhs_vector(bc.value)

                for i, I in enumerate(edge.dofmap):
                    f[I] += f_elem[i]
                # print(edge.nodes[0].coor, edge.nodes[1].coor)
                # print(f_elem)

        return f

    def get_closest_node_idx(self, coor):
        current_index = 0
        current_distance = None
        for n, i in enumerate(self.nodes):

            distance = sqrt((coor[0]-i.coor[0])*(coor[0]-i.coor[0])
                            + (coor[1]-i.coor[1])*(coor[1]-i.coor[1])
                            + (coor[2]-i.coor[2])*(coor[2]-i.coor[2]))

            if current_distance == None:
                current_distance = distance
            else:
                if distance < current_distance:
                    current_distance = distance
                    current_index = n

        # print(current_index, self.nodes[current_index].coor, current_distance)
        return current_index

    def solve(self, dirichlet_bcs, neumann_bcs=[], force_function=lambda x: [0.,0.]):

        # self.stiffness_matrix = csr_matrix(self.assemble_stiffness_matrix())
        self.stiffness_matrix = self.assemble_stiffness_matrix()

        self.neumann_rhs = self.assemble_neumann_rhs(neumann_bcs)
        self.force_rhs = self.assemble_force_rhs(force_function)
        self.rhs_vector_bc = self.neumann_rhs + self.force_rhs

        # self.stiffness_matrix_bc = self.stiffness_matrix.copy()
        self.stiffness_matrix_bc = np.array(self.stiffness_matrix)
        # self.rhs_vector_bc = np.array(self.rhs_vector)

        # for n in self.nodes:
        #     if n.coor[0]< 1e-12 and n.coor[1] < -0.5+1e-12:
        #         self.rhs_vector_bc[n.dofmap[1]] = -1.

        n_dof = self.stiffness_matrix.shape[0]

        for bc in dirichlet_bcs:
            for n in self.nodes:
                if not bc.position_bool(n.coor): continue

                value = bc.value(n.coor)
                for n,I in enumerate(n.dofmap):
                    for i in range(n_dof):
                        self.stiffness_matrix_bc[I,i] = 0.
                        self.stiffness_matrix_bc[i,I] = 0.

                    self.stiffness_matrix_bc[I,I] = 1.
                    self.rhs_vector_bc[I] = value[n]

        # import matplotlib
        # matplotlib.use('Qt4Agg')
        # import pylab as pl
        # pl.spy(self.stiffness_matrix_bc)
        # pl.show()

        logging.info('Attempting to solve %dx%d system'%(n_dof, n_dof))
        # self.solution_vector = np.linalg.solve(self.stiffness_matrix_bc, self.rhs_vector_bc)
        self.solution_vector = spsolve(self.stiffness_matrix_bc, self.rhs_vector_bc).reshape(self.rhs_vector_bc.shape)
        # import ipdb; ipdb.set_trace()
        self.rhs_vector = self.stiffness_matrix.dot(self.solution_vector)


        # force_resultant = [0.,0.]
        # for bc in neumann_bcs:
        #     for n in self.nodes:
        #         if not bc.position_bool(n.coor): continue

        #         value = bc.value(n.coor)
        #         for n,I in enumerate(n.dofmap):
        #             force_resultant[n] += self.rhs_vector[I,0]

        # print(force_resultant)

    def get_exact_solution_vector(self, exact):
        exact_solution_vector = np.zeros(self.solution_vector.shape)
        for n in self.nodes:
            exact_val = exact(n.coor)
            for n, dof in enumerate(n.dofmap):
                exact_solution_vector[dof] = exact_val[n]
        return exact_solution_vector

    def absolute_error(self, exact, exact_deriv, error='l2'):

        if error == 'l2':
            result = self.absolute_error_lp(exact, 2)
        elif error == 'linf':
            result = abs(self.solution_vector - self.get_exact_solution_vector(exact)).max()
        elif error == 'h1':
            l2 = self.absolute_error_lp(exact, 2)
            l2d = self.absolute_error_deriv_lp(exact_deriv, 2)
            result = pow(pow(l2,2.) + pow(l2d,2.), .5)
        else:
            raise Exception('Invalid error specification: %s'%error)

        return result

    def absolute_error_lp(self, exact, p):
        result = 0.

        for e in self.elems:
            coefficients = [self.solution_vector[i,0] for i in e.dofmap]
            result += e.absolute_error_lp(exact, coefficients, p)

        result = pow(result, 1./p)

        return result

    def absolute_error_deriv_lp(self, exact_deriv, p):
        result = 0.

        for e in self.elems:
            coefficients = [self.solution_vector[i,0] for i in e.dofmap]
            result += e.absolute_error_deriv_lp(exact_deriv, coefficients, p)

        result = pow(result, 1./p)

        return result

    def write_vtk(self, ofilename):
        logging.info('Writing %s'%ofilename)

        f = open(ofilename,'w')

        f.write("# vtk DataFile Version 3.1\n")
        f.write("Structural analysis of external ring fixator\n")
        f.write("ASCII\n")

        f.write("DATASET UNSTRUCTURED_GRID\n")
        # f.write("DATASET POLYDATA\n")

        n_points = len(self.nodes)
        n_cells = len(self.elems)
        f.write("POINTS  "+repr(n_points)+" FLOAT\n")

        for n in self.nodes:
            f.write("%.6e %.6e %.6e\n"%(n.coor[0],n.coor[1],0))

        f.write('\nCELLS %d %d\n'%(n_cells, n_cells * 5))
        for e in self.elems:
            f.write("4 ")
            for k in e.nodes:
                f.write("%d "%(k.idx))
            f.write("\n")

        f.write('\nCELL_TYPES '+repr(n_cells)+'\n')
        for e in range(n_cells):
            f.write('9 ')

        f.write('\n\nPOINT_DATA %d\n'%(n_points))

        f.write('VECTORS displacement float\n')
        for n, i in enumerate(self.nodes):
            for j in range(2):
                if i.dofmap[j] >= 0:
                    val = self.solution_vector[i.dofmap[j]]
                else:
                    val = 0.
                f.write('%.6e '%val)
            f.write('0.\n')
        f.write('\n')

        f.write('VECTORS force float\n')
        for n, i in enumerate(self.nodes):
            for j in range(2):
                if i.dofmap[j] >= 0:
                    val = self.rhs_vector[i.dofmap[j]]
                else:
                    val = 0.
                f.write('%.6e '%val)
            f.write('0.\n')
        f.write('\n')

        # f.write('CELL_DATA %d\n'%(n_cells))

        # f.write('VECTORS elem_local_force float\n')
        # for e in self.elems:
        #     f.write("%.6e %.6e %.6e\n"%(e.local_forces[0], e.local_forces[1], e.local_forces[2]))
        # f.write('\n')

        # f.write('VECTORS elem_local_moment float\n')
        # for e in self.elems:
        #     f.write("%.6e %.6e %.6e\n"%(e.local_moments[0], e.local_moments[1], e.local_moments[2]))
        # f.write('\n')

        # f.write('VECTORS elem_global_force float\n')
        # for e in self.elems:
        #     f.write("%.6e %.6e %.6e\n"%(e.global_forces[0], e.global_forces[1], e.global_forces[2]))
        # f.write('\n')

        # f.write('VECTORS elem_global_moment float\n')
        # for e in self.elems:
        #     f.write("%.6e %.6e %.6e\n"%(e.global_moments[0], e.global_moments[1], e.global_moments[2]))
        # f.write('\n')

        f.close()

