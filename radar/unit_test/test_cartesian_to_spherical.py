import unittest
import torch
import math
from gsplat.cuda._torch_impl_radar import _cartesian_to_spherical 
from gsplat.utils import upper_triangular_to_matrices

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
   
class TestCartesianToSpherical(unittest.TestCase):
    
    def test1(self):
        # Define input means and covariances
        
        # x, y, z
        means = torch.tensor([[10.0, 0.0, 0.0],
                              [10.0 * 1/math.sqrt(2), 10.0 * 1/math.sqrt(2), 0.0],
                              [0.0, 10.0, 0.0],
                              [10.0, 0.0, 0.0],
                              [10.0 * 1/math.sqrt(2), 10.0 * 1/math.sqrt(2), 0.0],
                              [0.0, 10.0, 0.0],
                              ])
        # xx, xy, xz, yy, yz, zz
        covars = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ])
        
        # Expected spherical means for the given Cartesian means
        # r, theta, phi
        expected_sph_means = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, torch.pi/4, 0.0],
            [10.0, torch.pi/2, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, torch.pi/4, 0.0],
            [10.0, torch.pi/2, 0.0],
        ], dtype=torch.float32)
        
        expected_sph_covars = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, math.atan2(1,10)**2],
            [0.0, 0.0, 0.0, 0.0, 0.0, math.atan2(1,10)**2],
            [0.0, 0.0, 0.0, 0.0, 0.0, math.atan2(1,10)**2],
            [0.0, 0.0, 0.0, math.atan2(1,10)**2, 0.0, 0.0],
            [0.5, 0.05, 0.0, 0.005, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
        return means, covars, expected_sph_means, expected_sph_covars
    
    def test2(self): # This will fail due to the nonlinearity in cart to spherical conversion
        # Define input means and covariances
        
        # x, y, z
        means = torch.tensor([[1.0, 0.0, 0.0],
                              [1.0 * 1/math.sqrt(2), 1.0 * 1/math.sqrt(2), 0.0],
                              [0.0, 1.0, 0.0]
                              ])
        # xx, xy, xz, yy, yz, zz
        covars = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        
        # Expected spherical means for the given Cartesian means
        # r, theta, phi
        expected_sph_means = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, torch.pi/4, 0.0],
            [1.0, torch.pi/2, 0.0],
        ], dtype=torch.float32)
        
        expected_sph_covars = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, math.atan2(1,1)**2],
            [0.0, 0.0, 0.0, 0.0, 0.0, math.atan2(1,1)**2],
            [0.0, 0.0, 0.0, 0.0, 0.0, math.atan2(1,1)**2],
        ], dtype=torch.float32)
        
        return means, covars, expected_sph_means, expected_sph_covars

    def test_cartesian_to_spherical(self):
        
        means, covars, expected_sph_means, expected_sph_covars = self.test1()

        # Run the function
        sph_means, sph_covars = _cartesian_to_spherical(means, covars)

        # Test spherical means
        torch.testing.assert_close(sph_means, expected_sph_means, atol=1e-4, rtol=1e-4)

        # Test spherical covariance
        torch.testing.assert_close(sph_covars, expected_sph_covars, atol=1e-4, rtol=1e-4)
        
        return

if __name__ == "__main__":
    unittest.main()

