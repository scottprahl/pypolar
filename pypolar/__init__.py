from importlib.metadata import version, PackageNotFoundError

__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2020-25, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/pypolar"

from .fresnel import *
from .jones import *
from .mueller import *
from .visualization import *

from .sym_fresnel import *
from .sym_jones import *
from .sym_mueller import *

from .ellipsometry import *

try:
    __version__ = version("pypolar")
except PackageNotFoundError:
    __version__ = "unknown"
