# Deve contenere il necessario per gestire i modelli (training, salvataggio, caricamento e valutazione)
# Si può anche inserire l'utilizzo di tflite, utile per valutare in cicli for
# Creare anche uno script per eseguire l'addestramento, che prende in inuput il file e salva il modello da qualche parte
# Bisogna creare un nuovo metodo in Cluster in cui si calcola la posizione con NN
# Bisogna aggiungere una nuova classe di ReconEvent in cui viene utilizzata la ricostruzione con NN

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import keras
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

from hexrec import HEXREC_MODELS

@dataclass
class ModelBase:
    """Base class for the definition of a neural network model
    """
    model: keras.models.Model

    def save(self, name: str) -> Path:
        """Save the model as a .keras file in the data directory

        Args:
            name (str): model name

        Returns:
            Path: absolute path of the save model
        """
        output_path = HEXREC_MODELS / f'{name}.keras'
        self.model.save(output_path)

        return output_path

    @classmethod
    def load(cls, name: str) -> 'ModelBase':
        """Load a saved model from the data directory

        Args:
            name (str): model name

        Returns:
            ModelBase: class instance
        """
        input_path = HEXREC_MODELS / f'{name}.keras'
        model = keras.models.load_model(input_path)

        return cls(model)

    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int,
              val_split: float = 0.2, **kwargs) -> None:
        
        self.model.fit(xdata, ydata, epochs=epochs, 
                       validation_split=val_split, **kwargs)

    def evaluate(self, xdata: np.ndarray):
        prediction = self.model({'input_layer': xdata})

        return  prediction.numpy()