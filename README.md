# pypolar

A basic collection of routines to track and visualize polarization 
through polarizers and birefringent elements.  Some basic ellipsometry
support is also included.

There are three modules
* pypolar.fresnel - Fresnel reflection and transmission calculations
* pypolar.jones   - Routines to support the Jones calculus
* pypolar.mueller - Routiens to support the Mueller calculus
	
Jupyter notebook documentation is included, but overall testing could be better.
The Mueller module is still incomplete.


## Usage

For examples, see the doc directory


## Installation

Use pip

    pip install pypolar

Alternatively you can install from github

    git clone https://github.com/scottprahl/pypolar.git

and add the pypolar directory to your PYTHONPATH


### Dependencies

Required Python modules: numpy, matplotlib


### License

pypolar is licensed under the terms of the MIT license.