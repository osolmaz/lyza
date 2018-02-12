from dolfin import *

from mshr import *
domain_vertices = [
    Point(0., 0., 0),
    Point(48., 44., 0),
    Point(48., 60., 0),
    Point(0., 44., 0),
]
polygon = Polygon(domain_vertices)
mesh = generate_mesh(polygon, 50)


# mesh = UnitSquareMesh(2, 2)
# mesh = UnitSquareMesh(32, 32)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0] < DOLFIN_EPS and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0] > 1 - DOLFIN_EPS and on_boundary

# Material parameters
lambda_ = 10000.0
mu = 1000.0

# Du = TrialFunction(V)
du = TrialFunction(V)
u  = TestFunction(V)

# The force on the right boundary
f  = Constant((0.0, 0.5))

ex  = Constant((1.0, 0.0))
ey  = Constant((0.0, 1.0))

I = Identity(len(u))

n = FacetNormal(mesh)


# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
AutoSubDomain(left).mark(boundary_subdomains, 1)
AutoSubDomain(right).mark(boundary_subdomains, 2)

dss = ds(subdomain_data=boundary_subdomains)

zero = Constant((0.0, 0.0))
bcs = DirichletBC(V, zero, left)

eps = sym(grad(u))
deps = sym(grad(du))

sigma = lambda_ * tr(eps) * I + 2 * mu * eps

a = inner(deps, sigma) * dx
L = inner(f, du) * dss(2)

u = Function(V)
solve(a == L, u, bcs)

f_ext_known = assemble(L)
f_ext_unknown = assemble(a)*u.vector() - f_ext_known


x_dofs = V.sub(0).dofmap().dofs()
y_dofs = V.sub(1).dofmap().dofs()

Fx_unknown = 0
Fy_unknown = 0

Fx_known = 0
Fy_known = 0


for i in x_dofs:
    Fx_unknown += f_ext_unknown[i][0]
    Fx_known += f_ext_known[i][0]

for i in y_dofs:
    Fy_unknown += f_ext_unknown[i][0]
    Fy_known += f_ext_known[i][0]

print("Reaction force:"+str([Fx_unknown, Fy_unknown]))
print("Given force   :"+str([Fx_known, Fy_known]))

file = File("out_fenics_cooks_membrane.pvd")
file << u


