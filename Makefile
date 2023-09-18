SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

default:
	@echo Type: make rcheck, make html, or make clean

check:
	-pyroma -d .
	-check-manifest
	make lintcheck
	make doccheck
	make notecheck

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck docs/jones-or-mueller.rst
	-rstcheck --ignore-directives automodule docs/pypolar.rst

lint:
	-pylint pypolar/gaertner.py
	-pylint pypolar/ellipsometry.py
	-pylint pypolar/fresnel.py
	-pylint pypolar/jones.py
	-pylint pypolar/mueller.py
	-pylint pypolar/sym_fresnel.py
	-pylint pypolar/sym_jones.py
	-pylint pypolar/sym_mueller.py
	-pylint pypolar/visualization.py

doccheck:
	-pydocstyle pypolar/gaertner.py
	-pydocstyle pypolar/ellipsometry.py
	-pydocstyle pypolar/fresnel.py
	-pydocstyle pypolar/jones.py
	-pydocstyle pypolar/mueller.py
	-pydocstyle pypolar/sym_fresnel.py
	-pydocstyle pypolar/sym_jones.py
	-pydocstyle pypolar/sym_mueller.py
	-pydocstyle pypolar/visualization.py

notecheck:
	make clean
	pytest --verbose tests/test_all_notebooks.py


html:
	cd docs && python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build
	open docs/_build/index.html

test:
	pytest tests/test_jones.py
	pytest tests/test_stokes.py

rcheck:
	make clean
	make lintcheck
	make doccheck
	make html
	make test
	-pyroma -d .
	-check-manifest
	make notecheck

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
	rm -rf pypolar.egg-info
	rm -rf pypolar/__pycache__
	rm -rf pypolar/__init__.pyc
	rm -rf pypolar/.ipynb_checkpoints
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf docs/omlc.org
	rm -rf docs/refractiveindex.info
	rm -rf docs/.ipynb_checkpoints
	rm -rf tests/__pycache__
.PHONY: clean check html test rcheck lintcheck doccheck notecheck