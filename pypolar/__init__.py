"""
================================================================================
pypolar: Analysis of polarization using the Jones and/or the Mueller calculus
================================================================================

    http://github.com/scottprahl/pypolar

Usage:

import pypolar.jones as jones
import pypolar.mueller as mueller

light = jones.field_horizontal()
print("Jones vector for horizontally-polarized light")
print(light)

light = mueller.field_left_circular()
print("Stokes vector for left circularly polarized light")
print(light)


"""

__author__ = 'Scott Prahl'
__version__ = '0.5.0'
