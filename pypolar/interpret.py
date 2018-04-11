"""
Simple routine for interpreting Jones vector

Todo:
    * handle cases when phase difference differs by more than 2pi
    * return angle for elliptical polarization
    * rename function

Scott Prahl
Apr 2018
"""
import numpy as np

__all__ = ['jones_check']


def jones_check(v):
    '''
    Interprets a Jones vector and returns a string describing the polarization state.

    Original by Alexander Miles 9/12/2013

    jones_check(v)

    Parameters
    ----------
    v     : A two element iterable containing the x, y Jones vector components, in that order.
               The components may be complex valued.

    Examples
    -------
    jones_check([1, -1j]) --> "Right circular polarization"

    jones_check([0.5, 0.5]) -->
                      "Linear polarization at 45.000000 degrees CCW from x-axis"

    jones_check( np.array([exp(-1j*pi), exp(-1j*pi/3)]) ) -->
                "Left elliptical polarization, rotated with respect to the axes"
    '''

    try:
        j1, j2 = v
    except:
        print("Jones vector must have two elements")
        return 0

    eps = 1e-12
    mag1, p1 = abs(j1), np.angle(j1)
    mag2, p2 = abs(j2), np.angle(j2)

    if np.remainder(p1 - p2, np.pi) < eps:
        ang = np.arctan2(mag2, mag1) * 180 / np.pi
        return "Linear polarization at %f degrees CCW from x-axis" % ang

    if abs(mag1 - mag2) < eps:
        if abs(p1 - p2 - np.pi / 2) < eps:
            s = "Right circular polarization"
        elif p1 > p2:
            s = "Right elliptical polarization, rotated with respect to the axes"
        if (p1 - p2 + np.pi / 2) < eps:
            s = "Left circular polarization"
        elif p1 < p2:
            s = "Left elliptical polarization, rotated with respect to the axes"
    else:
        if p1 - p2 == np.pi / 2:
            s = "Right elliptical polarization, non-rotated"
        elif p1 > p2:
            s = "Right elliptical polarization, rotated with respect to the axes"
        if p1 - p2 == -np.pi / 2:
            s = "Left circular polarization, non-rotated"
        elif p1 < p2:
            s = "Left elliptical polarization, rotated with respect to the axes"
    return s
