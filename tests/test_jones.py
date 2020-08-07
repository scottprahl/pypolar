import unittest
import numpy as np
import pypolar.jones as jones
import pypolar.mueller as mueller

class TestBasic(unittest.TestCase):

    def test_field_linear_H(self):
        H = jones.field_horizontal()
        J = jones.field_linear(0)
        for pair in zip(H, J):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_field_linear_V(self):
        V = jones.field_vertical()
        J = jones.field_linear(np.pi/2)
        for pair in zip(V, J):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_field_linear_multi(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        H = jones.field_horizontal()
        V = jones.field_vertical()
        J = jones.field_linear(angles)
        self.assertEqual(len(J),N)

    def test_intensity_scalar(self):
        J = jones.field_right_circular()
        I = jones.intensity(J)
        self.assertAlmostEqual(I,1)
        J = jones.field_horizontal()
        I = jones.intensity(J)
        self.assertAlmostEqual(I,1)

    def test_intensity_array(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        J = jones.field_linear(angles)
        I = jones.intensity(J)
        self.assertEqual(len(I),N)
        for intensity in I:
            self.assertAlmostEqual(intensity, 1)

    def test_phase_scalar(self):
        J = jones.field_left_circular()
        phi = jones.phase(J)
        self.assertAlmostEqual(phi, -np.pi/2)
        J = jones.field_right_circular()
        phi = jones.phase(J)
        self.assertAlmostEqual(phi, np.pi/2)
        J = jones.field_horizontal()
        phi = jones.phase(J)
        self.assertEqual(phi,0)

    def test_phase_array(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        J = jones.field_linear(angles)
        phi = jones.phase(J)
        self.assertEqual(len(phi),N)
        for p in phi:
            self.assertAlmostEqual(p, 0)

    def test_to_stokes_scalar(self):
        J = jones.field_left_circular()
        S = mueller.stokes_left_circular()
        SS = jones.jones_to_stokes(J)
        for pair in zip(S, SS):
            self.assertAlmostEqual(pair[0], pair[1])

    def test_to_stokes_array(self):
        N = 3
        angles = np.linspace(0,np.pi/2,N)
        J = jones.field_linear(angles)
        S = jones.jones_to_stokes(J)
        n,m = S.shape
        self.assertEqual(n,N)
        self.assertEqual(m,4)

if __name__ == '__main__':
    unittest.main()