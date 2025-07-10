"""Event reconstruction.
"""

import argparse
from tqdm import tqdm

from hexsample import logger
from hexsample.clustering import ClusteringNN
from hexsample.readout import HexagonalReadoutCircular
from hexsample.fileio import DigiInputFileCircular, ReconOutputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.recon import ReconEvent

__description__ = \
"""Run the reconstruction on a file produced by hxsim.py
"""

parser = argparse.ArgumentParser(description=__description__)
parser.add_argument('infile', help='Input file path')
parser.add_argument('--zsupthreshold', default=0, type=float,
                     help='zero suppresion threshold in ADC counts (default: 0)')
parser.add_argument('--nneighbors', default=6, type=int, 
                    help='number of neighbors to be considered (default: 6)')
parser.add_argument('--suffix', default='recon', type=str,
                    help='suffix for the output file (default: recon)')
args = parser.parse_args()

def hxrecon(args):
    """Application main entry point.
    """
    input_file_path = args.infile
    kwargs = {}
    kwargs['zsupthreshold'] = args.zsupthreshold
    kwargs['nneighbors'] = args.nneighbors
    kwargs['suffix'] = args.suffix

    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    logger.info(f'Readout chip: {readout}')

    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'])
    suffix = kwargs['suffix']
    output_file_path = input_file_path.replace('.h5', f'_{suffix}.h5')
    # ... and saved into an output file.
    output_file = ReconOutputFile(output_file_path)
    output_file.update_header(**kwargs)
    output_file.update_digi_header(**input_file.header)
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        args = event.trigger_id, event.timestamp(), event.livetime, cluster
        recon_event = ReconEvent(*args)
        mc_event = input_file.mc_event(i)
        output_file.add_row(recon_event, mc_event)
    output_file.flush()
    input_file.close()
    output_file.close()

    return output_file_path


if __name__ == '__main__':
    hxrecon(args)
