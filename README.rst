.. |pypi| image:: https://img.shields.io/pypi/v/pypolar?color=68CA66
   :target: https://pypi.org/project/pypolar/
   :alt: PyPI

.. |github| image:: https://img.shields.io/github/v/tag/scottprahl/pypolar?label=github&color=v
   :target: https://github.com/scottprahl/pypolar
   :alt: GitHub

.. |conda| image:: https://img.shields.io/conda/v/conda-forge/pypolar?label=conda&color=68CA66
   :target: https://github.com/conda-forge/pypolar-feedstock
   :alt: Conda

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8358111.svg
   :target: https://doi.org/10.5281/zenodo.8358111
   :alt: DOI

.. |license| image:: https://img.shields.io/github/license/scottprahl/pypolar?color=68CA66
   :target: https://github.com/scottprahl/pypolar/blob/main/LICENSE.txt
   :alt: License

.. |test| image:: https://github.com/scottprahl/pypolar/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/scottprahl/pypolar/actions/workflows/test.yaml
   :alt: Testing Status

.. |docs| image:: https://readthedocs.org/projects/pypolar/badge?color=68CA66
   :target: https://pypolar.readthedocs.io
   :alt: Documentation

.. |down| image:: https://img.shields.io/pypi/dm/pypolar?color=68CA66
   :target: https://pypi.org/project/pypolar/
   :alt: Download Count

.. |lite| image:: https://img.shields.io/badge/try-JupyterLite-68CA66.svg
   :target: https://scottprahl.github.io/pypolar/
   :alt: Try Online

pypolar
=======

|pypi| |github| |conda| |doi|

|license| |test| |docs| |down|

|lite|

``pypolar`` is a Python library for simulating, analyzing, and visualizing the polarization state of light as it propagates through optical systems. The package supports modeling with both Jones and Mueller calculus frameworks and includes functionality relevant to education, research, ellipsometry, and polarimetric system design.

The library provides computational tools, visualization utilities, and symbolic analysis support, making it suitable for laboratory instruction, computational optics coursework, and applied research in polarization optics.

----

Modules
-------

``pypolar`` is organized into several computational and symbolic components:

**Numerical computation modules**

* ``pypolar.fresnel`` — Fresnel reflection and transmission calculations

* ``pypolar.jones`` — Analysis of polarization using Jones calculus

* ``pypolar.mueller`` — Polarization modeling using the Mueller calculus

* ``pypolar.ellipsometry`` — Ellipsometry modeling tools

**Visualization support**

* ``pypolar.visualization`` — Poincaré sphere and vector-based visualization routines

**Symbolic computation**

* ``pypolar.sym_fresnel`` — Symbolic Fresnel reflection and transmission expressions

* ``pypolar.sym_jones`` — Symbolic polarization modeling using Jones calculus

* ``pypolar.sym_mueller`` — Symbolic Mueller matrix manipulation

----

Installation
============

``pypolar`` may be installed via ``pip``::

   pip install pypolar

or using ``conda``::

   conda install -c conda-forge pypolar

----

Documentation and Examples
===========================

Comprehensive user documentation, theory notes, and executable Jupyter examples are available at:

📄 https://pypolar.readthedocs.io

or use immediately in your browser via the JupyterLite button below

    |lite|

----

Example Usage
=============

The following example demonstrates modeling an optical isolator using the Jones formalism.

.. image:: https://raw.githubusercontent.com/scottprahl/pypolar/main/docs/isolator.png
  :width: 700px
  :alt: Optical isolator schematic

Jones Matrix Example
--------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pypolar.jones as jones
    import pypolar.visualization as vis
    
    J1 = jones.field_elliptical(np.pi/6, np.pi/6)
    J2 = jones.op_linear_polarizer(0) @ J1
    J3 = jones.op_quarter_wave_plate(np.pi/4) @ J2
    J4 = jones.op_mirror() @ J3
    J5 = jones.op_quarter_wave_plate(-np.pi/4) @ J4
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    vis.draw_empty_sphere(ax)
    
    vis.draw_jones_poincare(J1, ax, label='  start', color='red')
    vis.draw_jones_poincare(J2, ax, label='  after polarizer', color='blue')
    vis.draw_jones_poincare(J3, ax, label='  after QWP', color='blue')
    vis.draw_jones_poincare(J4, ax, label='  after mirror', color='blue')
    vis.draw_jones_poincare(J5, ax, label='  final', color='red')
    
    plt.show()

Mueller Matrix Example
----------------------

.. code-block:: python

    import numpy as np
    import pypolar.mueller as mueller
    
    A = mueller.stokes_right_circular()
    B = mueller.op_linear_polarizer(np.pi/4)
    C = mueller.op_quarter_wave_plate(0)
    D = mueller.op_mirror()
    E = mueller.op_quarter_wave_plate(0)
    F = mueller.op_linear_polarizer(-np.pi/4)
    F @ E @ D @ C @ B @ A

produces:

.. code-block:: python

    array([0., 0., 0., 0.])

----

Citation
--------

If you use ``pypolar`` in academic, instructional, or applied technical work, please cite:

Prahl, S. (2025). *pypolar: A Python module for polarization using Jones and Mueller calculus* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.8358111


BibTeX
^^^^^^

.. code-block:: bibtex

   @software{pypolar_prahl_2025,
     author    = {Scott Prahl},
     title     = {pypolar: A Python module for polarization using Jones and Mueller calculus},
     year      = {2025},
     version   = {1.0.1},
     doi       = {10.5281/zenodo.8358111},
     url       = {https://github.com/scottprahl/pypolar},
     publisher = {Zenodo}
   }

----

License
-------

``pypolar`` is distributed under the terms of the MIT License.
