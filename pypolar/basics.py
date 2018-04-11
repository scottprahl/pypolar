"""
Useful basic routines for managing polarization using the Jones calculu

Todo:
    * improve documentation of each routine

Scott Prahl
Apr 2018
"""

import numpy as np

__all__ = ['linear_polarizer',
           'retarder',
           'attenuator',
           'mirror',
           'rotation',
           'quarter_wave_plate',
           'half_wave_plate',
           'linear_polarized',
           'left_circular_polarized',
           'right_circular_polarized',
           'horizontal_polarized',
           'vertical_polarized',
           'intensity',
           'phase',
           'zero_if_near_zero']


def linear_polarizer(theta):
    """A linear polarized with axis at angle of theta horizontal plane"""

    return np.matrix([[np.cos(theta)**2, np.sin(theta) * np.cos(theta)],
                      [np.sin(theta) * np.cos(theta), np.sin(theta)**2]])


def retarder(theta, delta):
    """A half-wave plate with fast axis at angle of theta horizontal plane"""
    P = np.exp(+delta / 2 * 1j)
    Q = np.exp(-delta / 2 * 1j)
    D = np.sin(delta / 2) * 2j
    C = np.cos(theta)
    S = np.sin(theta)
    return np.array([[C * C * P + S * S * Q, C * S * D], [C * S * D, C * C * Q + S * S * P]])


def attenuator(od):
    """A neutral density filter with optical density od"""

    return np.matrix([[10**-od / 2, 0], [0, 10**-od / 2]])


def mirror():
    """Normal incidence on perfect mirror """
    return np.matrix([[1, 0], [0, -1]])


def rotation(theta):
    """Rotation by an angle theta"""
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


def quarter_wave_plate(theta):
    """A quarter-wave plate with fast axis at angle of theta horizontal plane"""

    return retarder(theta, np.pi / 2)


def half_wave_plate(theta):
    """A half-wave plate with fast axis at angle of theta horizontal plane"""

    return retarder(theta, np.pi)


def linear_polarized(theta):
    """Jones Vector corresponding to a linear polarization with angle with horizontal plane"""

    return np.array([np.cos(theta), np.sin(theta)])


def right_circular_polarized():
    """Jones Vector corresponding to right circular polarisation"""

    return 1 / np.sqrt(2) * np.array([1, -1j])


def left_circular_polarized():
    """Jones Vector corresponding to left circular polarisation"""

    return 1 / np.sqrt(2) * np.array([1, 1j])


def horizontal_polarized():
    """Jones Vector corresponding to horizontal polarized light"""

    return linear_polarized(0)


def vertical_polarized():
    """Jones Vector corresponding to horizontal polarized light"""

    return linear_polarized(np.pi / 2)


def normalize_vector(v):
    """
    Normalizes a vector by dividing each part by common number.
    After normalization the magnitude should be equal to ~1.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def intensity(v):
    """
    Returns the intensity
    """
    return np.abs(v[0])**2 + np.abs(v[1])**2


def phase(v):
    """
    Returns the phase
    """
    gamma = np.angle(v[1]) - np.angle([0])
    return gamma


def zero_if_near_zero(v):
    """
    Tries to remove near zero terms from value v
    """
    eps = 1e-9

    if abs(v.real) < eps:
        if abs(v.imag) < eps:
            return complex(0, 0)
        else:
            return complex(0, v.imag)
    else:
        if abs(v.imag) < eps:
            return complex(v.real, 0)
        else:
            return v


zero_if_near_zero = np.vectorize(zero_if_near_zero)
