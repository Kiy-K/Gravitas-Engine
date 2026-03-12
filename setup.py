"""
Build script for GRAVITAS Engine Cython extensions.

Usage:
    python setup.py build_ext --inplace
    # or via build_cython.sh
"""

import os
import sys
from setuptools import setup, Extension, find_packages

# ── Try Cython; fall back to pre-generated .c if unavailable ─────────────── #
USE_CYTHON = True
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
    cythonize = None
    from setuptools.command.build_ext import build_ext

import numpy as np


# ── Extension modules ────────────────────────────────────────────────────── #
# These are the performance-critical numerical kernels compiled to native C.

ext_suffix = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "gravitas_engine.core._kernels",
        sources=[os.path.join("gravitas_engine", "core", "_kernels" + ext_suffix)],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-march=native", "-funroll-loops", "-fno-math-errno"],
        extra_link_args=["-O3", "-lm"],
    ),
]

if USE_CYTHON:
    compiler_directives = {
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
        "language_level": 3,
        "embedsignature": True,
    }
    extensions = cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generate HTML annotation for profiling
    )


# ── Setup ─────────────────────────────────────────────────────────────────── #
setup(
    name="gravitas-engine",
    version="0.1.0",
    description="Research-grade governance RL environment",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "gymnasium>=0.29",
        "scipy>=1.10",
        "pyyaml>=6.0",
    ],
    setup_requires=[
        "cython>=3.0",
        "numpy>=1.24",
    ],
    zip_safe=False,
)
