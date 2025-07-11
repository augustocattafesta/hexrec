"""Extension of Source class from hexsample
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from hexsample.source import BeamBase, SpectrumBase

@dataclass
class TriangularBeam(BeamBase):
    """Triangular X-ray beam inside an hexagon 

    Arguments
    ---------
    x0 : float
        The x-coordinate of the center of the hexagon in cm.

    y0 : float
        The y-coordinate of the center of the hexagon in cm.

    A : np.ndarray
        The (x, y) coordinates of the first vertex of the hexagon in cm.

    B : np.ndarray
        The (x, y) coordinates of the second vertex of the hexagon in cm.
    """

    A_xy: tuple = (0, 0)
    B_xy: tuple = (0, 0)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        assert len(self.A_xy) == 2
        assert len(self.B_xy) == 2
        
        A = np.array([[*self.A_xy]])
        B = np.array([[*self.B_xy]])
        C = np.array([[self.x0, self.y0]])

        a = A - C
        b = B - C

        u = np.random.uniform(0, 1, (2, size))

        mask = u[0, :] + u[1, :] > 1
        u[:, mask] = 1 - u[:, mask]
        w = np.dot(a.T, u[0, :, None].T) + np.dot(b.T, u[1, :, None].T)

        return w[0] + C[0, 0], w[1] + C[0, 1]

class Line(SpectrumBase):
    """Class describing a monochromatic emission line at a given
    energy

    Args:
        SpectrumBase (_type_): _description_
    """

    def __init__(self, energy : float) -> None:
        """Constructor
        """
        self._energy = energy
        self._prob = 1

    def rvs(self, size = 1) -> np.ndarray:
        """

        Args:
            size (int, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: _description_
        """
        return np.full(size, self._energy)

    def plot(self) -> None:
        """Plot the monochromatic line
        """
        plt.bar(self._energy, self._prob, width=0.001, color='black')
        plt.xlabel('Energy [eV]')
        plt.ylabel('Relative intensity')
        plt.yscale('log')
        plt.grid()
