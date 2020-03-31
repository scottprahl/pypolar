from setuptools import setup

# use README as the long description
# make sure to use the syntax that works in both ReST and markdown
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type='text/x-rst'
)
# setup(
# 	long_description=
# 	"""
# 	A collection of routines for modeling polarization changes through
# 	birefringent elements and by reflection from surfaces.
# 	""",
# )
#
