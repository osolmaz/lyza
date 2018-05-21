from lyza_prototype.cell_processor import CellProcessor

class Integrator(CellProcessor):

    def integrate(self):
        n_dofs = len(self.mesh.nodes)*self.function_size
        result = 0.

        for idx, cell in enumerate(self.mesh.cells):
            if self.domain:
                pass
            else:
                if cell.is_boundary: continue

            elem_value = self.calculate_element_integral(cell)
            result += elem_value

        return result

