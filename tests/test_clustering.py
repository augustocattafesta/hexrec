"""Test of clustering.py
"""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

from hexsample.hexagon import HexagonalGrid, HexagonalLayout

from hexrec.clustering import Cluster



def test_cluster():
    pitch = 0.005
    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 10, 10, pitch)
    
    x0, y0 = 0, 0
    pix0 = grid.pixel_to_world(*grid.world_to_pixel(x0, y0))
    pix1 = grid.pixel_to_world(*grid.world_to_pixel(x0+pitch, y0))
    logger.debug(f'Pixel 0 position: {pix0}')
    logger.debug(f'Pixel 1 position: {pix1}')

    x = np.array([pix0[0], pix1[0]])
    y = np.array([pix0[1], pix1[1]])

    eta = np.linspace(0, 0.5, 100)
    yy = 0.5*(eta/0.5)**0.267

    # pulse_height = 100
    # pha = np.array([[1 - eta_i, eta_i] for eta_i in eta])*pulse_height

    # x_rc = np.zeros_like(eta)
    # x_bary = np.zeros_like(eta)

    # for i, p in enumerate(pha):
    #     cluster = Cluster(x, y, p)
    #     x_rc_i, _ = cluster.barycenter_1d(pitch)
    #     x_bary_i, _ = cluster.centroid()
    #     x_rc[i] = x_rc_i
    #     x_bary[i] = x_bary_i - pix0[0]

    # plt.plot(eta, yy, '-b')
    # plt.plot(eta, x_rc/pitch, '-k')
    # plt.plot(eta, x_bary/pitch, '-r')
    # plt.show()

    cluster = Cluster(x, y, np.array([90, 10]))
    print(cluster.fitted_position(), cluster.centroid())


if __name__ == '__main__':
    test_cluster()
