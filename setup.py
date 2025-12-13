from setuptools import setup, find_packages

setup(
    name="riemmpy",
    version="0.1.0",
    description="A symbolic differential geometry toolkit for general relativity and curved manifolds.",
    author="Edoardo Tesei",
    author_email="eddote6@protonmail.com",
    url="https://github.com/Joyboy0056/RiemPy",
    packages=find_packages(),
    install_requires=[
        "sympy",
        "numpy",
        "scipy",
        "matplotlib"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
