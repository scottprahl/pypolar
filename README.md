# pypolar

A basic collection of routines to track and visualize polarization 
through polarizers and birefringent elements.  Some basic ellipsometry
support is also included.

There are four modules
* pypolar.fresnel - Fresnel reflection and transmission calculations
* pypolar.jones   - Routines to support the Jones calculus
* pypolar.mueller - Routines to support the Mueller calculus
* pypolar.visualization - Routines to support visualization
	
Jupyter notebook documentation is included, but overall testing could be better.
The Mueller module is still incomplete.


### Simple Example

```python
import pypolar.mueller as mueller

# Optical Isolator example

A = mueller.stokes_right_circular()       # incident light
B = mueller.op_linear_polarizer(np.pi/4)  # polarizer at 45°
C = mueller.op_quarter_wave_plate(0)      # QWP with fast axis horizontal
D = mueller.op_mirror()                   # first surface mirror
E = mueller.op_quarter_wave_plate(0)      # QWP still has fast axis horizontal
F = mueller.op_linear_polarizer(-np.pi/4) # now at -45° because travelling backwards```

# net result is no light
F @ E @ D @ C @ B @ A
```

### Detailed Documentation

* [Jones Calculus](https://github.com/scottprahl/pypolar/blob/master/doc/01-jones.ipynb) 
* [Fresnel Reflection](https://github.com/scottprahl/pypolar/blob/master/doc/02-fresnel.ipynb) 
* [Mueller Calculus](https://github.com/scottprahl/pypolar/blob/master/doc/03-mueller.ipynb) 
* [Ellipsometry](https://github.com/scottprahl/pypolar/blob/master/doc/04-ellipsometry.ipynb) 
* [Fresnel Rhomb](https://github.com/scottprahl/pypolar/blob/master/doc/05-fresnel-rhomb.ipynb) 

## Installation

Use pip

    pip install pypolar


### Dependencies

Required Python modules: numpy, matplotlib


### License

pypolar is licensed under the terms of the MIT license.