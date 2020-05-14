from lyza.cell_iterator import CellIterator


class Integrator(CellIterator):
    def integrate(self):
        n_dofs = len(self.mesh.nodes) * self.function_size
        result = 0.0

        for idx, cell in enumerate(self.mesh.cells):
            if not self.domain.is_subset(cell):
                continue

            elem_value = self.calculate_element_integral(cell)
            result += elem_value

        return result
