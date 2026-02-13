"""pypolar: polarization analysis with Jones and Mueller calculus.

`pypolar` provides numerical and symbolic optics tools for:
- Jones vector/matrix modeling of coherent polarization states
- Mueller/Stokes modeling of partially polarized light
- Fresnel reflection/transmission operators and utility functions
- Ellipsometry calculations
- Visualization of fields, polarization ellipses, and Poincare states

The top-level package re-exports commonly used functions from:
- `pypolar.fresnel`, `pypolar.jones`, `pypolar.mueller`, `pypolar.ellipsometry`
- `pypolar.sym_fresnel`, `pypolar.sym_jones`, `pypolar.sym_mueller`
- `pypolar.visualization`

Common usage::

    import pypolar as pp
    J = pp.field_linear(0.0)
    M = pp.op_linear_polarizer(0.0)
    S = pp.jones_to_stokes(J)
"""

__version__ = "1.2.0"
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
