from sympy import symbols, sinh, Matrix
from sympy.diffgeom.rn import R2
from sympy.diffgeom import (
    metric_to_Christoffel_2nd,
    metric_to_Ricci_components,
    metric_to_Riemann_components,
    Manifold,
    Patch,
    CoordSystem,
    TensorProduct,
)

TP = TensorProduct
H2 = Manifold("H", 2)
patch = Patch("P", H2)

s, phi = symbols('s ϕ', real=True)
relation_dict = {
    # 'Car2D', 'Pol'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    # ('Pol', 'Car2D'): [(r, theta), (r*cos(theta), r*sin(theta))]
}
hyp2d_coords = CoordSystem("hyp2d", patch, (s, phi), relations=relation_dict)
ds, dphi = hyp2d_coords.base_oneforms()
# print(hyp2d_coords.base_scalars()) # s, ϕ
# print(hyp2d_coords.base_vectors()) # e_s, e_ϕ

g_hyperbolic = TP(ds, ds) + sinh(s)**2*TP(dphi, dphi)
print(metric_to_Ricci_components(g_hyperbolic))