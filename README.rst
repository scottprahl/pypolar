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

* ``pypolar.poincare`` — Dedicated Poincaré sphere plotting routines

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

Quickstart
==========

This short example combines numerical Jones/Mueller calculations with a
symbolic result.

.. code-block:: python

    import numpy as np
    import sympy
    import pypolar.jones as jones
    import pypolar.mueller as mueller
    import pypolar.sym_jones as sym_jones

    # Jones: left-circular light through a linear polarizer at 30 degrees
    J = jones.op_linear_polarizer(np.pi / 6) @ jones.field_left_circular()
    print("Jones output:", J)

    # Mueller: unpolarized input through the same polarizer
    S = mueller.op_linear_polarizer(np.pi / 6) @ mueller.stokes_unpolarized()
    print("Stokes output:", S)

    # Symbolic: Malus' law
    theta = sympy.symbols("theta", real=True)
    I = sympy.simplify(
        sym_jones.intensity(sym_jones.op_linear_polarizer(theta) * sym_jones.field_horizontal())[0]
    )
    print("Symbolic intensity:", I)


Documentation and Examples
===========================

Comprehensive user documentation, theory notes, and executable Jupyter examples are available at:

📄 https://pypolar.readthedocs.io

or use immediately in your browser via the JupyterLite button below

    |lite|

----

Examples
========

Circular Polarization Visualization
------------------------------------

.. code-block:: python

    from pypolar import jones
    from pypolar import visualization as vis

    v = jones.field_left_circular()
    print("Jones vector for left circularly polarized light:", v)
    ani = vis.draw_jones_animated(v, nframes=32)
    ani

will produce something like

.. image:: https://raw.githubusercontent.com/scottprahl/pypolar/main/docs/images/circular.gif
  :width: 700px
  :alt: Left circular polarization

----

Optical Isolator
----------------
.. image:: https://raw.githubusercontent.com/scottprahl/pypolar/main/docs/images/isolator.png
  :width: 700px
  :alt: Optical isolator schematic

The following example demonstrates modeling an optical isolator using the Jones formalism.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pypolar import jones
    from pypolar import visualization as vis

    b = jones.op_linear_polarizer(0)
    c = jones.op_quarter_wave_plate(np.pi / 4)
    d = jones.op_mirror()
    e = jones.op_quarter_wave_plate(-np.pi / 4)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    vis.draw_empty_sphere(ax)

    j1 = jones.field_elliptical(np.pi / 6, np.pi / 6)
    j2 = b @ j1
    j3 = c @ j2
    j4 = d @ j3
    j5 = e @ j4

    vis.draw_jones_poincare(j1, ax, label="  start", color="red", va="center")
    vis.draw_jones_poincare(j2, ax, label="  after Polarizer", color="blue", va="center")
    vis.draw_jones_poincare(j3, ax, label="  after QWP", color="blue", va="center")
    vis.draw_jones_poincare(j4, ax, label="  after mirror", color="blue", va="center")
    vis.draw_jones_poincare(j5, ax, label="  final", color="red", va="center")

    vis.join_jones_poincare(j1, j2, ax, color="blue", lw=2, linestyle=":")
    vis.join_jones_poincare(j2, j3, ax, color="blue", lw=2, linestyle=":")
    vis.join_jones_poincare(j3, j4, ax, color="blue", lw=2, linestyle=":")
    vis.join_jones_poincare(j4, j5, ax, color="blue", lw=2, linestyle=":")

    plt.show()

.. image:: https://raw.githubusercontent.com/scottprahl/pypolar/main/docs/images/poincare1.svg
  :width: 700px
  :alt: Poincare sphere

----

Symbolic Jones: Half-Wave Plate Rotation
----------------------------------------

This symbolic example verifies a useful identity: a half-wave plate with fast
axis angle ``theta`` rotates linear polarization from ``alpha`` to
``2*theta - alpha`` (up to a global phase factor, which does not affect the
physical polarization state). It also derives the analyzer transmission in
closed form.

.. code-block:: python

    import sympy
    import pypolar.sym_jones as sym_jones
    
    theta, alpha = sympy.symbols("theta alpha", real=True)
    
    J_in = sym_jones.field_linear(alpha)
    
    # Pass through a half-wave plate with fast axis at theta
    J_out = sympy.simplify(sym_jones.op_half_wave_plate(theta) * J_in)
    
    # Identity check using half wave plate
    J_expected = sympy.I * sym_jones.field_linear(2 * theta - alpha)
    print("Identity check:", sympy.simplify(J_out - J_expected))
    
    # Pass through a vertical analyzer and get intensity
    J = sym_jones.op_linear_polarizer(sympy.pi / 2) * J_out
    I = sym_jones.intensity(J)[0].simplify().trigsimp()
    print("I(theta, alpha) =", I)

produces:

.. code-block:: text

    Identity check: Matrix([[0], [0]])
    I(theta, alpha) = sin(alpha - 2*theta)**2

----

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

Prahl, S. (2026). *pypolar: A Python module for polarization using Jones and Mueller calculus* (Version 1.2.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.8358111


BibTeX
^^^^^^

.. code-block:: bibtex

   @software{pypolar_prahl_2026,
     author    = {Scott Prahl},
     title     = {pypolar: A Python module for polarization using Jones and Mueller calculus},
     year      = {2026},
     version   = {1.2.0},
     doi       = {10.5281/zenodo.8358111},
     url       = {https://github.com/scottprahl/pypolar},
     publisher = {Zenodo}
   }

----

License
-------

``pypolar`` is distributed under the terms of the MIT License.
