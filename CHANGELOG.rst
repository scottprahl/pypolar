Changelog
=================================================

Unreleased
-----------------
* add robust `interpret()` diagnostics across Jones and Mueller APIs:
  malformed input handling, unphysical-state checks, and consistent string returns
* expand symbolic Mueller interpretation parity with numeric behavior, including
  4x4 matrix admissibility diagnostics and zero-intensity guards in conversion paths
* make numeric/symbolic Fresnel utility surfaces symmetric and extend Mueller Fresnel
  operators with non-unity incident refractive index (`n_i`) support
* make neutral-density routines canonical around `op_attenuator()` with
  backward-compatible aliases
* add raw-component constructors for Jones and Mueller APIs:
  `field_components()`, `stokes_components()`, and symbolic counterparts
* add canonical Jones naming aliases:
  `ellipse_orientation()` and `ellipse_ellipticity()` in numeric Jones
* harden numeric Fresnel input handling:
  validate incidence-angle ranges, normalize positive-imaginary refractive indices
  by conjugation, and expand array/broadcast coverage
* normalize function and module docstrings across core modules to improve
  API discoverability and consistency
* broaden regression coverage across Jones/Mueller numeric and symbolic modules
  for edge cases, branch behavior, and API parity

1.1.0 (2/9/2026)
-----------------
* fix Jones ellipticity handedness branch-cut behavior for edge cases (including signed-zero sensitivity)
* extend Fresnel Jones operators to support non-unity incident refractive index `n_i`
* make Fresnel Jones transmission consistently a pure field-amplitude operator (including total internal reflection behavior)
* fix `mueller.stokes_elliptical()` broadcasting for mixed scalar/array inputs
* make numerical Fresnel Mueller reflection/transmission operators consistent with Jones-to-Mueller conversion
* fix `sym_jones` ellipse orientation/axes handling
* fix `sym_mueller.degree_of_polarization()` implementation
* make symbolic Fresnel Mueller transmission and reflection physically consistent with real-valued Mueller entries
* make `sym_mueller.op_fresnel_reflection()` support symbolic `m` and `theta` inputs
* add missing symbolic Stokes constructors in `sym_mueller` (`stokes_ellipsometry`, `stokes_elliptical`)
* align `sym_jones` symbolic API/docs and add symbolic conversion helpers (`jones_to_stokes`, `jones_op_to_mueller_op`)
* expand test coverage across `jones`, `sym_jones`, `mueller`, and `sym_mueller`, including a dedicated branch-focused test file for `jones.interpret()`
* broaden default CI/local test execution to run the full non-notebook test suite
* fix release metadata links for the PyPI workflow and project changelog URL
* update citation metadata release dates

1.0.2 (1/14/2026)
------------------
* support for jupyterlite
* remove requirements*.txt, all deps in pyproject.toml
* version info only in __init__.py
* update readthedocs configuration
* update docs/conf.py
* update github action to publish to pypi
* fix pylint warnings in update_citation.py
* move jupyter_lite_config.json to pypolar folder
* normalizing code formatting with black
* update to pypi-repository

1.0.1 (11/23/2025)
-------------------
* fix bug in calculation of t_s
* add tests for fresnel calculations
* fix __version__ so github actions work correctly

1.0.0 (11/23/2025)
-------------------
* support for jupyterlite
* use pyproject.toml and requirements-dev.txt
* better citation
* normalizing code formatting with black
* use venv for development consistency

0.9.3 (9/18/2023)
-----------------
* fixing zenodo

0.9.2 (9/18/2023)
-----------------
* automatic updating of pypi using github workflow

0.9.1 (9/18/2023)
-----------------
* include test files in tarball

0.9.0 (9/18/2023)
-----------------
* add citation with zenodo DOI
* add copyright to docs
* add conda support
* improve badges
* add automated github testing
* remove deprecated np.complex
* fix doc build
* add github auto testing
* lint files
* update refractive index urls in notebooks

v0.8.2 (9/5/2021)
-----------------
* do not import gaertner.py so pyserial is not a dependency

v0.8.1 (9/5/2021)
-----------------
* create pure python packages
* include wheel file
* package as python3 only
* allow degrees to be used
* add pypi badge
* remove unneeded shell commands
* remove unneeded pyserial dependency

v0.8.0
------
* documentation improvements
* automated notebook testing
* use sympy.I and sympy.pi

v0.7.0
------
* add Poincaré sphere support
* allow lists of Jones/Stokes vector to be processed
* add some basic tox tests

v0.6.0
------
* use sphinx for documentation
* host docs on https://pypolar.readthedocs.io
* remove unneeded files from pip installation
* start using tox for testing

v0.5.1
------
* fix jones.intensity() and sym_jones.intensity()

v0.5.0
------
* add symbolic routines
* add more mueller functions
* rename some stuff
* modify names for polarization ellipse
* improve readme
* split visualization into separate document

v0.3.1
------
* hopefully fix packaging
* fix phase for TIR of dielectrics

v0.3.0
------
* add basic routines for Mueller and Stokes vectors
* rename a bunch of routines
* repackage with the intent that modules will be used separately
* add Jupyter notebooks to provide basic documentation
* add basic ellipsometry support

v0.2.0
------
* add Fresnel transmission
* handle total internal reflection better

v0.1.1
------
* fix packaging error

v0.1.0
------
* initial checkin
