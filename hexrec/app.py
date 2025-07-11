"""Extension of app.py from hexsample"""

from hexsample.app import ArgumentParser
from hexrec.hexagon import HexagonalLayout

class ArgumentParser(ArgumentParser):
    def __init__(self, prog: str = None, usage: str = None, description: str = None) -> None:
        """Constructor"""
        super().__init__(prog, usage, description)

    def add_simple_readout_options(self) -> None:
        """Add an option group for the readout properties.
        """
        group = self.add_argument_group('readout', 'Redout configuration')
        layouts = [item.value for item in HexagonalLayout]
        group.add_argument('--layout', type=str, choices=layouts, default=layouts[0],
            help='hexagonal layout of the readout chip')
        group.add_argument('--numcolumns', type=int, default=304,
            help='number of colums in the readout chip (counting from 0 column)')
        group.add_argument('--numrows', type=int, default=352,
            help='number of rows in the readout chip (counting from 0 row)')
        group.add_argument('--pitch', type=float, default=0.005,
            help='pitch of the readout chip in cm')
        group.add_argument('--readoutmode', type=str, default='CIRCULAR',
            help='readout mode')
        group.add_argument('--noise', type=float, default=0.,
            help='equivalent noise charge rms in electrons')
        group.add_argument('--gain', type=float, default=1.,
            help='conversion factors between electron equivalent and ADC counts')
        group.add_argument('--zsupthreshold', type=int, default=0,
            help='zero-suppression threshold in ADC counts')
        
    def add_line_source_options(self) -> None:
        """Add an option group for line source properties.
        """
        group = self.add_argument_group('line source', 'X-ray source properties')
        group.add_argument('--energy', type=float, default=6000.,
                           help='energy of the monochromatic source')
        beams = ['gaussian', 'triangular']
        group.add_argument('--beamshape', type=str, choices=beams, default='gaussian',
                           help='X-ray beam morphology')
        group.add_argument('--srcposx', type=float, default=0.,
            help='x position of the source centroid or in cm')
        group.add_argument('--srcposy', type=float, default=0.,
            help='y position of the source centroid in cm')
        group.add_argument('--srcsigma', type=float, default=0.1,
            help='one-dimensional standard deviation of the gaussian beam in cm')
        group.add_argument('--trngindex', type=int, default=0,
            help='triangular section of the hexagon')

