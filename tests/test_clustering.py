"""Test of clustering.py
"""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

from hexsample.hexagon import HexagonalGrid, HexagonalLayout

from hexrec.clustering import Cluster

def test_cluster():
    """Test and compare the methods for estimate the reconstructed position.
    """
    # Should add also the neural network position.
    pitch = 0.005
    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 10, 10, pitch)

    x0, y0 = 0, 0
    pix0 = grid.pixel_to_world(*grid.world_to_pixel(x0, y0))
    pix1 = grid.pixel_to_world(*grid.world_to_pixel(x0+pitch, y0))
    logger.debug(f'Pixel 0 position: {pix0}')
    logger.debug(f'Pixel 1 position: {pix1}')

    x = np.array([pix0[0], pix1[0]])
    y = np.array([pix0[1], pix1[1]])

    pulse_height = 100
    eta = np.linspace(0, 0.5, 100)
    pha = np.array([[1-_eta, _eta] for _eta in eta])*pulse_height

    x_fit = np.zeros_like(eta)
    x_cen = np.zeros_like(eta)

    for i, _pha in enumerate(pha):
        cluster = Cluster(x, y, _pha, pitch=pitch, gamma=0.25)
        x_fit[i], _ = cluster.fitted_position()
        x_cen[i], _ = cluster.centroid()

    plt.plot(eta, (x_fit - x[0])/pitch, '-k', label='fit')
    plt.plot(eta, (x_cen - x[0])/pitch, '-b', label='centroid')
    plt.xlabel('eta')
    plt.ylabel('x_rc / pitch')
    plt.legend()

if __name__ == '__main__':
    test_cluster()
    plt.show()
