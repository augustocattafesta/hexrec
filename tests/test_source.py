"""Test for source.py"""

import numpy as np
import matplotlib.pyplot as plt

from loguru import logger

from hexrec.source import Line, TriangularBeam

from hexsample.hist import Histogram1d, Histogram2d
from hexsample.plot import setup_gca
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.display import HexagonalGridDisplay

def test_triangular_beam(num_photons = 1000000):
    """Test for TriangularBeam class

    Args:
        size (int, optional): _description_. Defaults to 10000.
    """
    beam = TriangularBeam(0, 0, (0, 1), (1, 0))

    x, y = beam.rvs(num_photons)
    binning_x = np.linspace(min(x), max(x), 100)
    binning_y = np.linspace(min(y), max(y), 100)
    plt.figure('Traingular beam')
    Histogram2d(binning_x, binning_y).fill(x, y).plot()
    setup_gca(xlabel='x [cm]', ylabel='y [cm]')
    plt.figure('Triangular beam x projection')
    hx = Histogram1d(binning_x).fill(x)
    hx.plot()
    plt.figure('Triangular beam x projection')
    hy = Histogram1d(binning_y).fill(y)
    hy.plot()

def test_triangular_beam_grid(num_photons = 100000):
    """Test for TrinagularBeam on hexagonal grid

    Args:
        num_photons (int, optional): _description_. Defaults to 100000.
    """
    target_col = 5
    target_row = 6
    layout = HexagonalLayout('ODD_Q')
    pitch = 0.005
    grid = HexagonalGrid(layout, 10, 8, pitch)
    x0, y0 = grid.pixel_to_world(target_col, target_row)
    logger.debug(f'Grid size -> {10, 8}')
    logger.debug(f'Target hexagon -> {target_col, target_row}')
    logger.debug(f'Target hexagon center -> {x0, y0}')
    display = HexagonalGridDisplay(grid)
    display.draw(pixel_labels=True)
    plt.plot(x0, y0, '+b')

    # Find vertex
    l = pitch / 2   # Not sure about this value, maybe different between layouts
    if grid.pointy_topped():
        A_xy = (x0 + l*np.sqrt(3)/2, y0 - l/2)
        B_xy = (x0 + l*np.sqrt(3)/2, y0 + l/2)
    else:
        A_xy = (x0 + l, y0)
        B_xy = (x0 + l/2, y0 + l*np.sqrt(3)/2)

    beam = TriangularBeam(x0, y0, A_xy, B_xy)
    x, y = beam.rvs(num_photons)
    plt.scatter(x, y, color='red', s=0.1)
    display.setup_gca()


    col, row = grid.world_to_pixel(x, y)
    pixel = np.array([col, row]).T
    pixel_vals = np.unique(pixel, return_counts=True, axis=0)

    logger.debug(f'Total number of photons: {num_photons}')
    for p_val, p_counts in zip(*pixel_vals):
        logger.debug(f'Pixel {tuple(p_val)} hit {p_counts}')

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
    # test_triangular_beam()
    test_triangular_beam_grid()
    # test_line(energy=7000)

    plt.show()
