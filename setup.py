from setuptools import setup, find_packages

setup(
    name="riempy",
    version="0.1.0",
    description="A symbolic Riemannian eometry toolkit for general relativity and curved manifolds.",
    author="Edoardo Tesei",
    url="https://github.com/Joyboy0056/RiemPy",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "sympy",
        "numpy",
        "scipy",
        "matplotlib"
    ],
    python_requires=">=3.8",
)