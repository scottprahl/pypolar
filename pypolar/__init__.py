"""pypolar: Polarization analysis and Jones/Mueller calculus for Python.

This package provides tools for:
- Jones vector and matrix calculations (coherent light)
- Mueller matrix and Stokes vector operations (partially polarized light)
- Fresnel equations for reflection and transmission
- Ellipsometry analysis
- Symbolic calculations using SymPy
- Visualization of polarization states

Author: Scott Prahl
License: MIT
"""

__version__ = "1.0.2"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2020-2026, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/pypolar"

# Numerical calculations
from .fresnel import *
from .jones import *
from .mueller import *
from .ellipsometry import *

# Symbolic calculations
from .sym_fresnel import *
from .sym_jones import *
from .sym_mueller import *

# Visualization
from .visualization import *
