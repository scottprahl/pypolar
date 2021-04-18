pypolar
=======

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/scottprahl/pypolar/blob/master

.. image:: https://img.shields.io/badge/readthedocs-latest-blue.svg
   :target: https://pypolar.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green.svg
   :target: https://github.com/scottprahl/pypolar

.. image:: https://img.shields.io/badge/MIT-license-yellow.svg
   :target: https://github.com/scottprahl/miepython/blob/master/LICENSE.txt

----

A collection of routines to track and visualize polarization
through polarizers and birefringent elements.  Some ellipsometry
support is also included.

There are four numeric modules

* `pypolar.fresnel` - reflection and transmission calculations
* `pypolar.jones` - management of polarization using the Jones calculus
* `pypolar.mueller` - management of polarization using the  Mueller calculus
* `pypolar.ellipsometry` - ellipsometry support

A module for visualization

* `pypolar.visualization` - Routines to support visualization

and three modules that support symbolic algebra

* `pypolar.sym_fresnel` - Fresnel reflection and transmission
* `pypolar.sym_jones` - Jones calculus
* `pypolar.sym_mueller` - Mueller calculus

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
