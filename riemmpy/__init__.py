"""
riempy: symbolic differential geometry in Python

Repository: https://github.com/Joyboy0056/RiemPy
Author: Edoardo Tesei (Joyboy0056)
"""

__version__ = "0.1.1"

from .geo_diff import (
    Manifold,
    Submanifold,
    Sphere,
    Hyp,
    Eucl,
    Minkowski,
    Schwarzschild
)

__all__ = [
    "Manifold",
    "Submanifold",
    "Sphere",
    "Hyp",
    "Eucl",
    "Minkowski",
    "Schwarzschild",
]

