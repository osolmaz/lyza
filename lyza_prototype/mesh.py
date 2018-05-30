from lyza_prototype.node import Node
from lyza_prototype.cell_quantity import CellQuantity
from lyza_prototype.function import Function
from lyza_prototype.domain import DefaultDomain
import time
import logging

class Mesh:

    def __init__(self):
        self.nodes = []
        self.cells = []
        self.boundary_cells = []
        self.quantitites = {}

        self.construct_mesh()

        for idx, n in enumerate(self.nodes):
            n.idx = idx

        for idx, c in enumerate(self.cells):
            c.idx = idx

    def construct_mesh(self):
        pass

    def add_node(self, coors):
        self.nodes.append(Node(coors, len(self.nodes)))

    def add_cell(self, cell):
        self.cells.append(cell)

    def get_n_nodes(self):
        return len(self.nodes)

    def set_quadrature_degree(
            self,
            quadrature_degree_map,
            spatial_dim,
            domain=DefaultDomain(),
            skip_basis=False):

        start = time.time()
        logging.debug('Started setting quadrature degree')

        # if not domain:
            # domain = DefaultDomain()

        quad_weight = CellQuantity(self, (1,1))
        quad_coor = CellQuantity(self, (3,1))

        for idx, cell in enumerate(self.cells):
            if not domain.is_subset(cell): continue

            degree = quadrature_degree_map(cell)
            quad_weights, quad_coors = cell.get_quad_points(degree)

            for weight in quad_weights:
                quad_weight.add_quantity_by_cell_idx(idx, weight)

            for coor in quad_coors:
                quad_coor.add_quantity_by_cell_idx(idx, coor)

        self.quantities = {
            'XL': quad_coor,
            'W': quad_weight,
        }

        if not skip_basis:
            N = CellQuantity(self, None)
            B = CellQuantity(self, None)
            J = CellQuantity(self, None)
            DETJ = CellQuantity(self, (1,1))
            JINVT = CellQuantity(self, None)
            XG = CellQuantity(self, (3, 1))

            for idx, cell in enumerate(self.cells):
                if not domain.is_subset(cell): continue

                degree = quadrature_degree_map(cell)

                N_arr, B_arr, J_arr, DETJ_arr, JINVT_arr, XG_arr \
                    = cell.calculate_basis_values(spatial_dim, degree)

                for i in N_arr: N.add_quantity_by_cell_idx(idx, i)
                for i in B_arr: B.add_quantity_by_cell_idx(idx, i)
                for i in J_arr: J.add_quantity_by_cell_idx(idx, i)
                for i in DETJ_arr: DETJ.add_quantity_by_cell_idx(idx, i)
                for i in JINVT_arr: JINVT.add_quantity_by_cell_idx(idx, i)
                for i in XG_arr: XG.add_quantity_by_cell_idx(idx, i)

        self.quantities = {
            **self.quantities,
            'N': N,
            'B': B,
            'J': J,
            'DETJ': DETJ,
            'JINVT': JINVT,
            'XG': XG,
        }

        logging.debug('Finished setting quadrature degree in %fs'%(time.time()-start))

    def get_position_function(self, spatial_dimension):
        if spatial_dimension > 3:
            raise Exception()

        result = Function(self, spatial_dimension)

        for i, n in enumerate(self.nodes):
            for j in range(spatial_dimension):
                result.vector[i*spatial_dimension+j] = n.coor[j]

        return result

    def init_quantity(self, key, shape):
        result = CellQuantity(self, shape)

        for cell in self.cells:
            n_array = len(self.quantities['W'].get_quantity(cell))
            result.add_zero_array(cell, n_array=n_array)

        self.quantities[key] = result


