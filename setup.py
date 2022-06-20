from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

# Extensions' configuration

library_dirs = ["/usr/local/lib"]

extra_compile_args = [
    "-ffast-math",
    "-march=native",
    "-msse2"
]

include_dirs = [
    "/usr/local/include",
    numpy.get_include()
]

# Extension modules

ext_modules = [
    Extension(
        name="hplop.core.equations",
        sources=["src/hplop/core/equations.pyx"],
        libraries=["cspice", "m"],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args
    ),
    Extension(
        name="hplop.solver.rkn1210",
        sources=["src/hplop/solver/rkn1210.pyx"],
        libraries=["m"],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args
    ),
    Extension(
        name="hplop.solver.rkn1210_coeffs",
        sources=["src/hplop/solver/rkn1210_coeffs.pyx"],
        libraries=["m"],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args
    ),
    Extension(
        name="hplop.pck.pck",
        sources=["src/hplop/pck/*.pyx"],
    )
]

# Perform setup

if __name__ == "__main__":

    setup(
        include_dirs=include_dirs,
        ext_modules=cythonize(ext_modules, annotate=True)
    )
