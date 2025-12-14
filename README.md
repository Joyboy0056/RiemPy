# RiemPy

> A symbolic toolkit for differential geometry and general relativity in Python.

RiemPy is a Python library for modeling Riemannian manifolds and computing fundamental geometric quantities in a symbolic way. It is designed for students, researchers, and enthusiasts of general relativity and differential geometry.
## 🚀 Installation

Clone the repository and install in dev mode:

```bash
git clone https://github.com/Joyboy0056/RiemPy.git
cd RiemPy
pip install -e .
```
or directly install the package from the public repo:
```bash
pip install git+https://github.com/Joyboy0056/RiemPy.git
```

### Note
This is an early version of the library — useful, but still a draft.  
Expect changes, improvements, and possibly breaking updates in future releases.

---

## Quick start

Please refer to the examples in this repository or see the code below and use it as a reference:

```bash
from riemmpy import Manifold, Submanifold, Sphere, Schwarzschild
from sympy import pprint, Matrix, exp, symbols

pprint(Sphere(2).metric)

x, y, z, t = symbols('x y z t')
R3 = Manifold(Matrix.eye(3), [x, y, z])

S2 = Submanifold(R3, Sphere(2).coords, Sphere(2).get_embedding())
S2.get_induced_metric()
pprint(Sphere(2).metric==S2.metric)

pprint(Schwarzschild(4).horizon.metric)

# Godel metric
g = Matrix([
    [1,      0,     exp(x),            0],
    [0,     -1,      0,            0],
    [exp(x),     0, -1/2 * exp(x)**2,      0],
    [0,      0,      0,           -1]
])

Godel = Manifold(g, [x, y, z, t])

Godel.get_scalar_curvature()
pprint(Godel.scalar_curvature)
```

## 📦 Module Documentation

Built on top of `sympy`, the library allows users to compute geometric quantities such as Christoffel symbols, Ricci tensor, curvature scalars and geodesics.

### Class: `Manifold`

A base class representing a general (possibly pseudo) Riemannian manifold of dimension n.

**Attributes:**
- `metric`: The metric tensor expressed as a `sympy.Matrix`.
- `coords`: A list of symbolic coordinate variables.

**Main methods:**
- `get_christoffel_symbols()`: Computes the Christoffel symbols of the metric.
- `get_riemann_tensor()`: Computes the Riemann curvature tensor.
- `get_ricci_tensor()`: Computes the Ricci curvature tensor.
- `get_ricci_scalar()`: Computes the Ricci scalar.
- `get_einstein_tensor()`: Computes the Einstein tensor.
- `is_flat()`: Checks whether the Riemann tensor vanishes identically.
- `is_einstein_mfd()`: Checks whether the manifold is of Einstein type.
- `print_sectional_curvatures()`: Prints the sectional curvatures of the metric.
- `get_kretschmann_scalar()`: Computes the Kretschmann scalar.
- `create_spacetime_metric(scale_factor)`: Constructs the Lorentzian metric from a pure Riemannian, with `scale_factor` as a `sympy` function.
- `display_geodesic_equations()`: Computes and print the geodesic equations symbolically.
- `solve_geodesic()`: Solves the geodesic equations numerically.
- `get_geometrics()`: Computes all the main geometric quantities of the manifold.

**Minor methods:**
The `Manifold` class has been provided with some other minor methods, useful for computations of functions on manifolds, as `covariant_derivative(X, T, ind)`, `gradient(f)`, `divergence(X)`, `laplacian(f)`, `hessian(f)`.


---

### Class: `Submanifold`

A subclass of `Manifold` representing a (n-k)-dimensional submanifold.

**Attributes:**
- `ambient_manifold`: A `Manifold` object.
- `sub_coords`: A list of symbolic coordinate variables.
- `embedding`: A list of symbolic functions of the coordinates.

**Main methods:**
- `get_embedding_jacobian()`: Computes the Jacobian matrix of the embedding map.
- `get_induced_metric()`: Computes the induced metric from the embedding.
- `get_normal_field()`: Computes the normal vector field.
- `get_IInd_fundamental_form()`: Computes the extrinsic curvature tensor.
- `get_mean_curvatureII()`: Computes the extrinsic curvature scalar.
- `is_minimal()`: Checks whether the mean curvature scalar vanishes identically.
- `is_totally_geodesic()`: Checks whether the mean curvature tensor vanishes identically.
- `plot_geodesics_on_surface(domain, geodesics)`: For a 2-dimensional surface, plot the geodesics on the surface.
- `get_geometrics_sub()`: Computes all the main geometric quantities of the submanifold.

The subclass `Submanifold` inherits the methods of `Manifold`. 
Here below, you can find other classes for the main geometric spaces.

---

### Class: `Sphere`

A subclass for the spherical n-dimensional space S^n.

**Attributes:**
- `dimension`: An integer for the dimension of the manifold.

**Main methods:**
Being a `Manifold` objects it inherits the base class methods. Moreover
- `get_embedding()`: Returns the embedding of S^n in R^n+1.

---

### Class: `Hyp`

A subclass for the hyperbolic n-dimensional space H^n.

**Attributes:**
- `dimension`: An integer for the dimension of the manifold.

**Main methods:**
Being a `Manifold` objects it inherits the base class methods.


---

### Class: `Eucl`

A subclass for the Euclidean n-dimensional space R^n.

**Attributes:**
- `dimension`: An integer for the dimension of the manifold.

**Main methods:**
Being a `Manifold` objects it inherits the base class methods.

**Instances:**
- `polar`: A `Manifold` object for the polar representation of `Eucl(n)`.

---

### Class: `Minkowski`

A subclass for the Minkowskian (n+1)-dimensional space R^(1,n).

**Attributes:**
- `neg`: An integer for the negative part of the signature.
- `pos`: An integer for the positive part of the signature.

**Main methods:**
Being a `Manifold` objects it inherits the base class methods.


### Class: `Schwarzschild`

A subclass for the spacelike Schwarzschild n-dimensional space Sigma^n.

**Attributes:**
- `dimension`: An integer for the dimension of the manifold.

**Main methods:**
Being a `Manifold` objects it inherits the base class methods.

**Instances:**
- `spacetime`: A `Manifold` object for the (n+1)-dimensional Schwarzschild spacetime.
- `horizon`: A `Submanifold` object for the (n-1)-dimensional event horizon.



