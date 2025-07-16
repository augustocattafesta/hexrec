"""Extension of hexsample hexagon module"""

from typing import Tuple

import numpy as np

from hexsample.hexagon import HexagonalLayout, HexagonalGrid

class HexagonalGrid(HexagonalGrid):
    """Extension of HexagonalGrid from hexsample
    """
    def __init__(self, layout: HexagonalLayout, num_cols: int, num_rows: int,
                 pitch: float) -> None:
        super().__init__(layout, num_cols, num_rows, pitch)

    @staticmethod
    def create_rotator(theta: float):
        c = np.cos(theta)
        s = np.sin(theta)
        def rotate(v):
            x, y = v[0], v[1]
            x_rot = x*c - y*s
            y_rot = x*s + y*c

            return np.array([x_rot, y_rot])

        return rotate

    def find_vertices(self, col: np.array, row: np.array,
                      i: int = 0) -> Tuple[np.array, np.array, np.array]:
        """Find the vertices of a triangular section of hexagons. The first section,
        corresponding to i=0, is the one which lies on the x-axis or intersects it,
        depending on the orientation of the hexagon. The sections are counted 
        counter-clockwise.

        Args:
            col (np.array): array with column pixel coordinates of hexagons
            row (np.array): array with row pixel coordinates of hexagons
            i (int, optional): triangular section. Defaults to 0.

        Returns:
            Tuple[np.array, np.array, np.array]: the first array are the (x, y) coordinates
            of the centers, the second and the third contain each the (x, y) coordinates of
            one of the other two vertices of the triangle 
        """
        x0, y0 = self.pixel_to_world(col, row)
        a = np.array([x0, y0]).T

        size = self.pitch / np.sqrt(3)
        rotator = self.create_rotator(i*np.pi/3)

        if self.pointy_topped():
            v0 = rotator(np.array([size*np.sqrt(3)/2, -size/2]))
            v1 = rotator(np.array([size*np.sqrt(3)/2, size/2]))
        else:
            v0 = rotator(np.array([size, 0]))
            v1 = rotator(np.array([size/2, size*np.sqrt(3)/2]))

        b = a + v0
        c = a + v1

        return a, b, c
