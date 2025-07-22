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
        beams = ['gaussian', 'triangular', 'hexagonal']
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

    def add_analysis_options(self) -> None:
        """Add an option group for the analysis.
        """
        group = self.add_argument_group('analysis', 'Options for analysis')
        group.add_argument('--bins', type=int, default=100,
                           help='number of bins for histogram analysis')
        # Could add save plot option

    def add_eta_options(self) -> None:
        """Add an option group for eta function analysis.
        """
        group = self.add_argument_group('analysis', 'Options for eta function analysis')
        group.add_argument('--npixels', type=int, default=2,
                           help='dimension of the cluster for the analysis')
        group.add_argument('--zsupthreshold', type=int, default=30,
            help='zero-suppression threshold in ADC counts')
        group.add_argument('--nneighbors', type=int, default=6,
            help='number of neighbors to be considered (0--6)')
        group.add_argument('--bins', type=int, default=20,
            help='number of bins for the profile of eta')

    def add_reconstruction_options(self) -> None:
        """Add an option group for reconstruction.
        """
        group = self.add_argument_group('reconstruction', 'Options for event reconstruction')
        group.add_argument('--zsupthreshold', type=int, default=30,
            help='zero-suppression threshold in ADC counts')
        group.add_argument('--nneighbors', type=int, default=6,
            help='number of neighbors to be considered (0--6)')
        group.add_argument('--rcmethod', choices=['centroid', 'fit', 'nnet'], type=str,
            default='centroid', help='How to reconstruct position')
        group.add_argument('--gamma', default=0.272, type=float,
            help='index of the power law for position fit')
        group.add_argument('--nnmodel', default=None, type=str,
            help='model to use for reconstruction with neural network')
        group.add_argument('--suffix', default='recon', type=str,
                    help='suffix for the output file')

    def add_model_name(self) -> None:
        self.add_argument('nnmodel', type=str, default='model',
                          help='name of the neural network model')

    def add_neural_net_options(self) -> None:
        """Add an option group for neural network
        """
        group = self.add_argument_group('nnetwork', 'Options for event neural network training')
        group.add_argument('--npixels', type=int, default=-1,
            help='cluster size of events to analyze, -1 means all events')
        group.add_argument('--epochs', type=int, default=10,
            help='number of epochs for training')
