pypolar
=======

A basic collection of routines to track and visualize polarization
through polarizers and birefringent elements.  Some basic ellipsometry
support is also included.

There are four numeric modules

* `pypolar.fresnel <https://github.com/scottprahl/pypolar/blob/master/pypolar/fresnel.py>`_ - Fresnel reflection and transmission calculations
* `pypolar.jones <https://github.com/scottprahl/pypolar/blob/master/pypolar/jones.py>`_ - Routines to support the Jones calculus
* `pypolar.mueller <https://github.com/scottprahl/pypolar/blob/master/pypolar/mueller.py>`_ - Routines to support the Mueller calculus
* `pypolar.visualization <https://github.com/scottprahl/pypolar/blob/master/pypolar/visualization.py>`_ - Routines to support visualization

and three modules that support symbolic algebra

* `pypolar.sym_fresnel <https://github.com/scottprahl/pypolar/blob/master/pypolar/sym_fresnel.py>`_ - Symbolic Fresnel reflection and transmission
* `pypolar.sym_jones <https://github.com/scottprahl/pypolar/blob/master/pypolar/sym_jones.py>`_  - Symbolic Jones calculus
* `pypolar.sym_mueller <https://github.com/scottprahl/pypolar/blob/master/pypolar/sym_mueller.py>`_ - Symbolic Mueller calculus

and finally a module for ellipsometry

* `pypolar.ellipsometry <https://github.com/scottprahl/pypolar/blob/master/pypolar/ellipsometry.py>`_ - Basic ellipsometry

Detailed documentation is available at `Read the Docs <https://pypolar.readthedocs.io>`_.

Installation
------------

Use pip::

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

To Do
-----

* Overall testing could be better.
*  The Mueller module is still incomplete.

License
-------

pypolar is licensed under the terms of the MIT license.
