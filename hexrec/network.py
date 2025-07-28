# Deve contenere il necessario per gestire i modelli (training, salvataggio, caricamento e valutazione)
# Si può anche inserire l'utilizzo di tflite, utile per valutare in cicli for


from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import math

import numpy as np
import keras
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

from hexrec import HEXREC_MODELS

class ModelBase:
    """Base class for the definition of a neural network model
    """

    def save(self, name: str) -> Path:
        """Save the model in "models" folder 

        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> 'ModelBase':
        """Load the model at the given path
        
        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError
    
    @classmethod
    def load_pretrained(cls) -> 'ModelBase':
        """Load the pre-trained model of the package
        
        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError
    
    def train(self, *args) -> None:
        """Train the neural network

        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError
        
    def predict(self, xdata: np.ndarray) -> np.ndarray:
        """Generate the predicted output for the input sample
        
        This needs to be overloaded by any derived classes.
        """
        raise NotImplementedError

@dataclass
class ModelDNN(ModelBase):
    """Class to manage Deep Dense Network
    """
    model: keras.models.Model

    def save(self, name: str) -> Path:
        """Overloaded method
        """
        output_path = HEXREC_MODELS / f'{name}.keras'
        self.model.save(output_path)

        return output_path

    @classmethod
    def load(cls, path: str) -> 'ModelBase':
        """Overloaded method
        """
        model = keras.models.load_model(path)

        return cls(model)

    @classmethod
    def load_pretrained(cls) -> 'ModelBase':
        """Overloaded method
        """
        with resources.path('hexrec.models', 'modelDNN.keras') as model_path:
            return cls.load(str(model_path))

    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int,
              val_split: float = 0.2, **kwargs) -> None:
        
        return self.model.fit(xdata, ydata, epochs=epochs, 
                       validation_split=val_split, **kwargs)

    def predict(self, xdata: np.ndarray):
        input_layer_name = self.model.layers[0].name
        prediction = self.model({input_layer_name: xdata})

        return  prediction.numpy()

class ModelGNN(ModelBase):
    """Class to manage Graph Neural Network 
    """
    model: torch.nn.Module

    def save(self, name: str) -> Path:
        """Overloaded method
        """
        output_path = HEXREC_MODELS / f'{name}.pt'
        torch.save(self.model.state_dict(), output_path)
        return output_path

    def load(self, path: str) -> None:
        """Overloaded method
        """
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def event_to_graph(xdata: np.ndarray, ydata: np.ndarray = None) -> Data:
        """Convert event data to a graph of PyTorch Geometric

        Args:
            pha (np.ndarray): normalized pha array with values of all of the 7 pixels
            x (np.ndarray): normalized x-position of all of the 7 pixels
            y (np.ndarray): normalized y-position of all of the 7 pixels

        Returns:
            Data: graph of the event
        """
        assert xdata.shape == (7, 3)

        nodes = torch.tensor(xdata, dtype=torch.float)

        
        edge_list = [[0, i] for i in range(1, 7)] + [[i, 0] for i in range(1, 7)]
        positions = xdata[:, 1:]
        for i in range(1, 7):
            for j in range(1, 7):
                if i < j:
                    dist = math.sqrt(np.sum((positions[i] - positions[j])**2))
                    if abs(dist - 1) < 1e-5:
                        edge_list.extend([[i, j], [j, i]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        if ydata is not None:
            y = torch.tensor(ydata, dtype=torch.float).unsqueeze(0)
        else:
            y = None

        data = Data(x=nodes, edge_index=edge_index, y=y)

        return data
    
    @staticmethod
    def load_data(xdata: np.ndarray, ydata: np.ndarray = None, **kwargs) -> DataLoader:
            dataset = [__class__.event_to_graph(_xdata, _ydata) for _xdata, _ydata in zip(xdata, ydata)]
            dataset_loader = DataLoader(dataset, **kwargs)

            return dataset_loader


        
