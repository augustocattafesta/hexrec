"""Test for hexagon.py"""

import numpy as np
import matplotlib.pyplot as plt

from hexsample.display import HexagonalGridDisplay
from hexsample.plot import plt

from hexrec.hexagon import HexagonalLayout, HexagonalGrid

def test_find_vertices():
    """Test find_vertices method
    """
    target_cols = np.array([0, 1, 2, 3])
    target_rows = np.array([0, 0, 0, 0])

    layout = HexagonalLayout('ODD_R')
    pitch = 0.005
    grid = HexagonalGrid(layout, 10, 8, pitch)
    display = HexagonalGridDisplay(grid)
    display.draw(pixel_labels=True)
    a, b, c = grid.find_vertices(target_cols, target_rows, i = 6)

    # Plot centers
    plt.plot(a[:, 0], a[:, 1], '+r')
    # Plot vertices
    plt.plot(b[:, 0], b[:, 1], '+b')
    plt.plot(c[:, 0], c[:, 1], '+g')

    display.setup_gca()

if __name__ == '__main__':
    test_find_vertices()
    plt.show()
