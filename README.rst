pypolar
=======

A collection of routines to track and visualize polarization
through polarizers and birefringent elements.  Some ellipsometry
support is also included.

There are four numeric modules

* `pypolar.fresnel` - Fresnel reflection and transmission calculations
* `pypolar.jones` - Routines to support the Jones calculus
* `pypolar.mueller` - Routines to support the Mueller calculus
* `pypolar.ellipsometry` - Routines to support ellipsometry

A module for visualization

* `pypolar.visualization` - Routines to support visualization

and three modules that support symbolic algebra

* `pypolar.sym_fresnel` - Symbolic Fresnel reflection and transmission
* `pypolar.sym_jones` - Symbolic Jones calculus
* `pypolar.sym_mueller` - Symbolic Mueller calculus

Detailed documentation is available at `Read the Docs <https://pypolar.readthedocs.io>`_.

Installation
------------

Use `pip`::

    pip install pypolar

Usage
-----

Create an optical isolator::

    import pypolar.mueller as mueller

    # Optical Isolator example, no light returning

    A = mueller.stokes_right_circular()       # incident light
    B = mueller.op_linear_polarizer(np.pi/4)  # polarizer at 45°
    C = mueller.op_quarter_wave_plate(0)      # QWP with fast axis horizontal
    D = mueller.op_mirror()                   # first surface mirror
    E = mueller.op_quarter_wave_plate(0)      # QWP still has fast axis horizontal
    F = mueller.op_linear_polarizer(-np.pi/4) # now at -45° because travelling backwards

    F @ E @ D @ C @ B @ A

License
-------

pypolar is licensed under the terms of the MIT license.
