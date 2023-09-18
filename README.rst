pypolar
=======

by Scott Prahl

.. image:: https://img.shields.io/pypi/v/pypolar.svg
   :target: https://pypi.org/project/pypolar/
   :alt: pypi

.. image:: https://img.shields.io/conda/v/conda-forge/pypolar.svg
   :target: https://github.com/conda-forge/pypolar-feedstock
   :alt: conda

.. image:: https://zenodo.org/badge/107437651.svg
   :target: https://zenodo.org/badge/latestdoi/107437651
   :alt: zenodo

|

.. image:: https://img.shields.io/badge/MIT-license-yellow.svg
   :target: https://github.com/scottprahl/pypolar/blob/master/LICENSE.txt
   :alt: License

.. image:: https://github.com/scottprahl/pypolar/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/pypolar/actions/workflows/test.yml
   :alt: testing

.. image:: https://readthedocs.org/projects/pypolar/badge
  :target: https://pypolar.readthedocs.io
  :alt: docs

.. image:: https://img.shields.io/pypi/dm/pypolar
   :target: https://pypi.org/project/pypolar/
   :alt: Downloads

----

Code to model and visualize the polarization state of light as it travels
through polarizers and birefringent elements.  Some ellipsometry
support is also included.

There are four numeric modules:

* `pypolar.fresnel` - reflection and transmission calculations
* `pypolar.jones` - management of polarization using the Jones calculus
* `pypolar.mueller` - management of polarization using the  Mueller calculus
* `pypolar.ellipsometry` - ellipsometry support

A module for visualization:

* `pypolar.visualization` - Routines to support visualization

and three modules that support symbolic algebra:

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
