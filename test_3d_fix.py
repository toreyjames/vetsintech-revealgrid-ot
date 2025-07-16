#!/usr/bin/env python3
"""
Test script to verify matplotlib 3D projection fix.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def test_3d_projection():
    """Test that 3D projection works correctly."""
    try:
        # Create a simple 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate sample data
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        # Create 3D surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_title('3D Projection Test - Success!')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.savefig('test_3d_success.png')
        plt.close()
        
        print("✅ 3D projection test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ 3D projection test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_3d_projection() 