import unittest
import numpy as np
import pypolar.jones as jones
import pypolar.mueller as mueller

class TestBasic(unittest.TestCase):

    def test_field_linear_H(self):
        H = mueller.stokes_horizontal()
        S = mueller.stokes_linear(0)
        for pair in zip(H, S):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_field_linear_V(self):
        V = mueller.stokes_vertical()
        S = mueller.stokes_linear(np.pi/2)
        for pair in zip(V, S):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_field_linear_multi(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        H = mueller.stokes_horizontal()
        V = mueller.stokes_vertical()
        S = mueller.stokes_linear(angles)
        self.assertEqual(len(S),N)

    def test_intensity_scalar(self):
        S = mueller.stokes_right_circular()
        I = mueller.intensity(S)
        self.assertAlmostEqual(I,1)
        S = mueller.stokes_horizontal()
        I = mueller.intensity(S)
        self.assertAlmostEqual(I,1)

    def test_intensity_array(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        S = mueller.stokes_linear(angles)
        I = mueller.intensity(S)
        self.assertEqual(len(I),N)
        for intensity in I:
            self.assertAlmostEqual(intensity, 1)

    def test_dop_scalar(self):
        S = mueller.stokes_left_circular()
        dop = mueller.degree_of_polarization(S)
        self.assertAlmostEqual(dop, 1)
        S = mueller.stokes_unpolarized()
        print(S)
        dop = mueller.degree_of_polarization(S)
        self.assertAlmostEqual(dop, 0)
        S = mueller.stokes_elliptical(0.5, np.pi/6, np.pi/3)
        dop = mueller.degree_of_polarization(S)
        self.assertAlmostEqual(dop, 0.5)

    def test_dop_array(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        S = mueller.stokes_linear(angles)
        dop = mueller.degree_of_polarization(S)
        self.assertEqual(len(dop),N)
        for p in dop:
            self.assertAlmostEqual(p, 1)

    def test_to_jones_scalar(self):
        S = mueller.stokes_left_circular()
        J = jones.field_left_circular()
        JJ = mueller.stokes_to_jones(S)
        for pair in zip(J, JJ):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_to_stokes_array(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        S = mueller.stokes_linear(angles)
        J = mueller.stokes_to_jones(S)
        n,m = J.shape
        self.assertEqual(n,N)
        self.assertEqual(m,2)

if __name__ == '__main__':
    unittest.main()