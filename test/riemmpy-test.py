from sympy import symbols, Matrix, sinh, pprint
from riemmpy import Manifold

# Poincaré half plane
s, phi = symbols('s ϕ')
g_hyperbolic = Matrix([[1, 0], [0, sinh(s)**2]])
H2 = Manifold(g_hyperbolic, [s, phi], name="H")

H2.get_geometrics()

print("Ricci Tensor")
pprint(H2.ricci_tensor)
print("\nScalar Curvature:", H2.scalar_curvature)
print()
H2.print_sectional_curvatures()

print(H2)
print()
print(repr(H2))