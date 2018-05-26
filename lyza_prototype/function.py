import numpy as np

class Function:
    def __init__(self, mesh, function_size):
        # self.function_space = function_space
        self.function_size = function_size
        self.mesh = mesh

        n_dof = function_size*len(mesh.nodes)
        self.vector = np.zeros((n_dof,1))
        self.label = None

        self.node_dofs = []
        for n in self.mesh.nodes:
            self.node_dofs.append([n.idx*function_size+i for i in range(function_size)])

    def set_vector(self, vector):
        self.vector = vector

    def set_label(self, label):
        self.label = label

    def get_node_val(self, idx):
        start_idx = idx*self.function_size
        end_idx = idx*self.function_size + self.function_size
        return self.vector[start_idx:end_idx]

    def copy(self):
        result = Function(self.function_space)
        result.vector = self.vector.copy()
        return result

    def set_analytic_solution(self, function, time=0):

        for n in self.mesh.nodes:
            analytic_val = function(n.coor, time)
            for n, dof in enumerate(self.node_dofs[n.idx]):
                self.vector[dof] = analytic_val[n]

