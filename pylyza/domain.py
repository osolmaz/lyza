

# class Domain:

#     def __init__(self, mesh, subdomain_function=None):
#         self.mesh = mesh
#         self.subdomain_func = None
#         # pass

#     def get_subdomain(self, subdomain_func):
#         return Domain(mesh, subdomain_function=subdomain_function)
#         # for i in self.mesh.elems:

#     def get_n_nodes(self):
#         return len(self.mesh.nodes)

#     def get_finite_elements(self, function_space):
#         result = []
#         if self.subdomain_func:
#             for c in self.mesh.cells:
#                 if subdomain_func(c, False):
#                     result.append(c.get_finite_element(function_space))
#             for c in self.mesh.edges:
#                 if subdomain_func(c, True):
#                     result.append(c.get_finite_element(function_space))
#         else:
#             for c in self.mesh.cells:
#                 result.append(c.get_finite_element(function_space))
#         return result


# class Subdomain:
#     def __init__(self, subdomain_function):
#         self.mesh = mesh
#         self.subdomain_func = None

#     def get_subdomain(self, subdomain_func):
#         return Domain(mesh, subdomain_function=subdomain_function)

