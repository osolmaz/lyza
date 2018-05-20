from lyza_prototype.node import Node
from lyza_prototype.cell_quantity import CellQuantity

class Mesh:

    def __init__(self):
        self.nodes = []
        self.cells = []
        self.boundary_cells = []

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

    def get_quadrature_quantities(self, quadrature_degree_map):
        quad_weight_quantity = CellQuantity(self, (1,1))
        quad_coor_quantity = CellQuantity(self, (3,1))

        for idx, cell in enumerate(self.cells):
            degree = quadrature_degree_map(cell)
            quad_weights, quad_coors = cell.get_quad_points(degree)

            for weight in quad_weights:
                quad_weight_quantity.add_quantity_by_cell_idx(idx, weight)

            for coor in quad_coors:
                quad_coor_quantity.add_quantity_by_cell_idx(idx, coor)

        return quad_weight_quantity, quad_coor_quantity

    def get_basis_quantities(self, quadrature_degree_map, spatial_dim):
        N = CellQuantity(self, (1,1))
        B = CellQuantity(self, (spatial_dim,1))
        jac = CellQuantity(self, (spatial_dim, 1))
        det_jac = CellQuantity(self, (1,1))
        jac_inv_tra = CellQuantity(self, (1,spatial_dim))
        global_coor = CellQuantity(self, (3, 1))

        for idx, cell in enumerate(self.cells):
            degree = quadrature_degree_map(cell)

            N_arr, B_arr, jac_arr, det_jac_arr, jac_inv_tra_arr, global_coor_arr \
                = cell.calculate_basis_values(spatial_dim, degree)

            for i in N_arr: N.add_quantity_by_cell_idx(idx, i)
            for i in B_arr: B.add_quantity_by_cell_idx(idx, i)
            for i in jac_arr: jac.add_quantity_by_cell_idx(idx, i)
            for i in det_jac_arr: det_jac.add_quantity_by_cell_idx(idx, i)
            for i in jac_inv_tra_arr: jac_inv_tra.add_quantity_by_cell_idx(idx, i)
            for i in global_coor_arr: global_coor.add_quantity_by_cell_idx(idx, i)

        return N, B, jac, det_jac, jac_inv_tra, global_coor

