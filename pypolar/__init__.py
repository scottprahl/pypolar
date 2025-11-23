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

__version__ = "1.0.1"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2020-25, Scott Prahl"
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

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__copyright__",
    "__license__",
    "__url__",
]
