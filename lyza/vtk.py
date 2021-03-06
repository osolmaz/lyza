import logging
import lyza.cells as cells


class VTKFile:
    def __init__(self, path):
        self.path = path

    def calculate_n_cell_data(self, mesh):
        result = 0
        for e in mesh.cells:
            if e.is_boundary:
                continue
            result += len(e.nodes) + 1

        return result

    def write(self, mesh=None, functions=[]):
        if not mesh and not functions:
            raise Exception("Input either a mesh or a function")
        elif not mesh:
            mesh = functions[0].mesh

        if not isinstance(functions, list):
            functions = [functions]

        logging.info("Writing %s" % self.path)

        f = open(self.path, "w")

        f.write("# vtk DataFile Version 3.1\n")
        f.write("LYZA Output\n")
        f.write("ASCII\n")

        f.write("DATASET UNSTRUCTURED_GRID\n")
        # f.write('DATASET POLYDATA\n')

        n_points = len(mesh.nodes)
        n_cells = len([i for i in mesh.cells if not i.is_boundary])
        f.write("POINTS  " + repr(n_points) + " FLOAT\n")

        for n in mesh.nodes:
            f.write("%.6e %.6e %.6e\n" % (n.coor[0], n.coor[1], n.coor[2]))

        f.write("\nCELLS %d %d\n" % (n_cells, self.calculate_n_cell_data(mesh)))
        for e in mesh.cells:
            if e.is_boundary:
                continue
            if isinstance(e, cells.Quad):
                f.write("4 ")
            elif isinstance(e, cells.Hex):
                f.write("8 ")
            else:
                raise Exception("Invalid cell for VTK file")

            for k in e.nodes:
                f.write("%d " % (k.idx))
            f.write("\n")

        f.write("\nCELL_TYPES " + repr(n_cells) + "\n")
        for e in mesh.cells:
            if e.is_boundary:
                continue
            if isinstance(e, cells.Quad):
                f.write("9 ")
            elif isinstance(e, cells.Hex):
                f.write("12 ")
            else:
                raise Exception("Invalid cell for VTK file")

        if functions:
            f.write("\n\nPOINT_DATA %d\n" % (n_points))

            for function in functions:
                # V = function.function_space

                if function.mesh != mesh:
                    raise Exception("Function does not match input mesh")

                if not function.label:
                    raise Exception("Function is not labeled")

                dim = function.function_size

                if dim == 1:
                    f.write("SCALARS %s float\n" % function.label)
                    f.write("LOOKUP_TABLE default\n")

                    for n, i in enumerate(mesh.nodes):
                        val = function.vector[function.node_dofs[n][0]]
                        f.write("%.6e\n" % val)
                    f.write("\n")

                elif dim == 2 or dim == 3:
                    f.write("VECTORS %s float\n" % function.label)

                    for n, i in enumerate(mesh.nodes):
                        vector = [0.0, 0.0, 0.0]
                        for j, k in enumerate(function.node_dofs[n]):
                            vector[j] = function.vector[k]

                        f.write("%.6e %.6e %.6e\n" % (vector[0], vector[1], vector[2]))
                    f.write("\n")
                elif dim > 3:
                    f.write("FIELD FieldData 1\n")
                    f.write("%s %d %d float\n" % (function.label, dim, n_points))

                    for n, i in enumerate(mesh.nodes):
                        for j, k in enumerate(function.node_dofs[n]):
                            f.write("%.6e" % function.vector[k])
                            if j < dim - 1:
                                f.write(" ")
                            elif j == dim - 1:
                                f.write("\n")
                    f.write("\n")

                else:
                    raise Exception()

        # f.write('CELL_DATA %d\n'%(n_cells))

        # f.write('VECTORS elem_local_force float\n')
        # for e in self.elems:
        #     f.write('%.6e %.6e %.6e\n'%(e.local_forces[0], e.local_forces[1], e.local_forces[2]))
        # f.write('\n')

        f.close()
