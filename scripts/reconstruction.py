"""Event reconstruction.
"""

from tqdm import tqdm

from hexsample import logger
from hexsample.readout import HexagonalReadoutCircular
from hexsample.fileio import DigiInputFileCircular, ReconOutputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.recon import ReconEvent

from hexrec.app import ArgumentParser
from hexrec.clustering import ClusteringNN
from hexrec.recon import ReconEventFitted, ReconEventNNet
from hexrec.network import ModelBase, ModelDNN, ModelGNN

__description__ = \
"""Run the reconstruction on a file produced by hxsim.py
"""

PARSER = ArgumentParser(description=__description__)
PARSER.add_infile()
PARSER.add_reconstruction_options()
PARSER.add_nnet_recon_options()

def hxrecon(**kwargs):
    """Application main entry point.
    """
    input_file_path = kwargs['infile']
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    logger.info(f'Readout chip: {readout}')

    model = None
    if kwargs['rcmethod'] == 'dnn':
        if kwargs['nnmodel'] == 'pretrained':
            model = ModelDNN.load_pretrained()
        elif kwargs['nnmodel'] == 'custom':
            if kwargs['modelname'] is not None:
                model = ModelDNN.load(kwargs['modelname'])
            else:
                raise RuntimeError('insert a custom model name --modelname MODELNAME')
    elif kwargs['rcmethod'] == 'gnn':
        if kwargs['nnmodel'] == 'pretrained':
            model = ModelGNN.load_pretrained()
        else:
            raise RuntimeError('only pretrained model available')

    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'],
                              header['pitch'], kwargs['gamma'], model=model)
    suffix = kwargs['suffix']
    output_file_path = input_file_path.replace('.h5', f'_{suffix}.h5')
    # ... and saved into an output file.
    output_file = ReconOutputFile(output_file_path)
    output_file.update_header(**kwargs)
    output_file.update_digi_header(**input_file.header)

    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        if kwargs['npixels'] == -1 or cluster.size() == kwargs['npixels']:
            args = event.trigger_id, event.timestamp(), event.livetime, cluster
            if kwargs['rcmethod'] == 'centroid':
                recon_event = ReconEvent(*args)
            elif kwargs['rcmethod'] == 'fit':
                recon_event = ReconEventFitted(*args)
            elif kwargs['rcmethod'] == 'dnn' or kwargs['rcmethod'] == 'gnn':
                recon_event = ReconEventNNet(*args)
            else:
                raise RuntimeError

            mc_event = input_file.mc_event(i)
            output_file.add_row(recon_event, mc_event)

    output_file.flush()
    input_file.close()
    output_file.close()

    return output_file_path

if __name__ == '__main__':
    hxrecon(**vars(PARSER.parse_args()))
