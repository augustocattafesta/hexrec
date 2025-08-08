"""Extension of Source class from hexsample
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from hexsample.source import BeamBase, SpectrumBase
from hexrec.hexagon import HexagonalGrid

@dataclass
class TriangularBeam(BeamBase):
    """Triangular X-ray beam inside an hexagon 

    Arguments
    ---------
    x0 : float
        The x-coordinate of the center of the hexagon in cm.

    y0 : float
        The y-coordinate of the center of the hexagon in cm.

    v0 : np.ndarray
        The (x, y) coordinates of the first vertex of the hexagon in cm.

    v1 : np.ndarray
        The (x, y) coordinates of the second vertex of the hexagon in cm.
    """

    v0: tuple = (0, 0)
    v1: tuple = (0, 0)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        assert len(self.v0) == 2
        assert len(self.v1) == 2

        v0_array = np.array([[*self.v0]])
        v1_array = np.array([[*self.v1]])
        center = np.array([[self.x0, self.y0]])

        a = v0_array - center
        b = v1_array - center

        u = np.random.uniform(0, 1, (2, size))

        mask = u[0, :] + u[1, :] > 1
        u[:, mask] = 1 - u[:, mask]
        w = np.dot(a.T, u[0, :, None].T) + np.dot(b.T, u[1, :, None].T)

        x = w[0] + center[0, 0]
        y = w[1] + center[0, 1]

        return x, y

@dataclass
class HexagonalBeam(BeamBase):
    """Triangular X-ray beam inside an hexagon 

    Arguments
    ---------
    x0 : float
        The x-coordinate of the center of the hexagon in cm.

    y0 : float
        The y-coordinate of the center of the hexagon in cm.

    v0 : np.ndarray
        The (x, y) coordinates of the first vertex of the hexagon in cm.

    v1 : np.ndarray
        The (x, y) coordinates of the second vertex of the hexagon in cm.

    """
    v0: tuple = (0, 0)
    v1: tuple = (0, 0)

    def rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Overloaded method.

        Arguments
        ---------
        size : int
            The number of X-ray photon positions to be generated.

        Returns
        -------
        x, y : 2-element tuple of np.ndarray of shape ``size``
            The photon positions on the x-y plane.
        """
        _, size_t = np.unique(np.random.randint(0, 6, size), return_counts=True)
        x = np.zeros(size)
        y = np.zeros(size)

        j = 0
        c = np.array([self.x0, self.y0])
        for i, t_s in enumerate(size_t):
            rotator = HexagonalGrid.create_rotator(np.pi/3*i)
            v0_rot = rotator((self.v0[0] - c[0], self.v0[1] - c[1])) + c
            v1_rot = rotator((self.v1[0] - c[0], self.v1[1] - c[1])) + c
            beam = TriangularBeam(self.x0, self.y0, tuple(v0_rot), tuple(v1_rot))
            x_tr, y_tr = beam.rvs(t_s)

            x[j:j+t_s] = x_tr
            y[j:j+t_s] = y_tr
            j += t_s

        return x, y

class Line(SpectrumBase):
    """Class describing a monochromatic emission line at a given
    energy
    """

    def __init__(self, energy : float) -> None:
        """Constructor of the class
        """
        self._energy = energy
        self._prob = 1

    def rvs(self, size = 1) -> np.ndarray:
        """

        Args:
            size (int, optional): The number of X-ray photon energies to be generated. 
            Defaults to 1.

        Returns:
            energy (np.ndarray): The photon energies in eV.
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
