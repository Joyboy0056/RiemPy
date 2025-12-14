"""
riempy: symbolic differential geometry in Python

Repository: https://github.com/Joyboy0056/RiemPy
Author: Edoardo Tesei (Joyboy0056)
"""

__version__ = "0.1.1"

from .geo_diff.manifold import Manifold
from .geo_diff.submanifolds import (
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

