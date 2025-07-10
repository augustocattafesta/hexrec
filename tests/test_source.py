"""Test for source.py"""

import numpy as np
import matplotlib.pyplot as plt

from loguru import logger

from hexrec.source import Line

def test_line(energy = 6000, size = 100000):
    """Test for line class

    Args:
        energy (int, optional): energy of the beam. Defaults to 6000.
        size (int, optional): number of events. Defaults to 100000.
    """
    line = Line(energy)
    logger.debug(line)
    x = line.rvs(size)

    values, counts = np.unique(x, return_counts=True)
    logger.debug(f'Beam energy: {energy}')
    logger.debug(f'Number of events: {size}')
    for val, cnts in zip(values, counts):
        logger.debug(f'{val} eV -> {cnts} counts')
        # Check if all events have the same energy given as input
        assert val == energy
        assert cnts == size

    line.plot()

if __name__ == '__main__':
    test_line(energy=7000)

    plt.show()
