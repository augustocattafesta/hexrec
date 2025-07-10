"""Extension of Source class from hexsample
"""

import numpy as np
import matplotlib.pyplot as plt

from hexsample.source import SpectrumBase

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
