"""Extension of recon.py from hexsample
"""

from dataclasses import dataclass
from typing import Tuple

from hexsample.recon import DEFAULT_IONIZATION_POTENTIAL

from hexrec.clustering import Cluster

@dataclass
class ReconEventFitted:

    """Descriptor for a reconstructed event with only two pixels.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    timestamp : float
        The timestamp (in s) of the event.

    livetime : int
        The livetime (in us) since the last event.

    roi_size : int
        The ROI size for the event.

    cluster : Cluster
        The reconstructed cluster for the event.
    """

    trigger_id: int
    timestamp: float
    livetime: int
    #roi_size: int
    cluster: Cluster

    def energy(self, ionization_potential: float = DEFAULT_IONIZATION_POTENTIAL) -> float:
        """Return the energy of the event in eV.

        .. warning::
           This is currently using the ionization energy of Silicon to do the
           conversion, assuming a detector gain of 1. We will need to do some
           bookkeeping, here, to make this work reliably.
        """
        return ionization_potential * self.cluster.pulse_height()

    def position(self) -> Tuple[float, float]:
        """Return the reconstructed position of the event.
        """
        return self.cluster.fitted_position()

@dataclass
class ReconEventNNet:

    """Descriptor for a reconstructed event with only two pixels.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    timestamp : float
        The timestamp (in s) of the event.

    livetime : int
        The livetime (in us) since the last event.

    roi_size : int
        The ROI size for the event.

    cluster : Cluster
        The reconstructed cluster for the event.
    """

    trigger_id: int
    timestamp: float
    livetime: int
    #roi_size: int
    cluster: Cluster

    def energy(self, ionization_potential: float = DEFAULT_IONIZATION_POTENTIAL) -> float:
        """Return the energy of the event in eV.

        .. warning::
           This is currently using the ionization energy of Silicon to do the
           conversion, assuming a detector gain of 1. We will need to do some
           bookkeeping, here, to make this work reliably.
        """
        return ionization_potential * self.cluster.pulse_height()

    def position(self) -> Tuple[float, float]:
        """Return the reconstructed position of the event.
        """
        return self.cluster.nnet_position()

