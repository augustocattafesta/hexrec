# Deve contenere il necessario per gestire i modelli (training, salvataggio, caricamento e valutazione)
# Si può anche inserire l'utilizzo di tflite, utile per valutare in cicli for


from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import math
from typing import Tuple

from loguru import logger
import numpy as np
import keras
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

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
    def load(cls, path: str) -> 'ModelDNN':
        """Overloaded method
        """
        model = keras.models.load_model(path)

        return cls(model)

    @classmethod
    def load_pretrained(cls) -> 'ModelDNN':
        """Overloaded method
        """
        with resources.path('hexrec.models', 'modelDNN.keras') as model_path:
            return cls.load(str(model_path))

    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int,
              val_split: float = 0.2, **kwargs) -> None:
        
        return self.model.fit(xdata, ydata, epochs=epochs, 
                       validation_split=val_split, **kwargs)

    def predict(self, xdata: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_layer_name = self.model.layers[0].name
        prediction = self.model({input_layer_name: xdata})
        prediction = prediction.numpy()

        return  prediction[:, 0], prediction[:, 1]

@dataclass
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

    @classmethod
    def load_pretrained(cls) -> 'ModelGNN':
        """Overloaded method
        """
        model = GNNRegression()
        instance = cls(model=model)
        with resources.path('hexrec.models', 'modelGNN.pt') as model_path:
            instance.load(str(model_path))
            return instance

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
    def data_loader(xdata: np.ndarray, ydata: np.ndarray = None, batch_size: int = 16, **kwargs) -> DataLoader:
        """Prepare data for training or prediction

        Args:
            xdata (np.ndarray): data for training or prediction. Shape must be (7, 3) for single data, or (N, 7, 3)
                for multiple data
            ydata (np.ndarray, optional): data for training. Shape must be (2,) for single data
                or (N, 2) for multiple data

        Returns:
            DataLoader: _description_
        """
        if len(xdata.shape) == 2:
            assert xdata.shape == (7, 3)
            dataset = [ModelGNN.event_to_graph(xdata, ydata)]
        
        elif len(xdata.shape) == 3:
            assert xdata.shape[1:] == (7, 3)
            if ydata is not None:
                dataset = [ModelGNN.event_to_graph(_xdata, _ydata) for _xdata, _ydata in zip(xdata, ydata)]
            else:
                dataset = [ModelGNN.event_to_graph(_xdata) for _xdata in xdata]

        dataset_loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

        return dataset_loader

    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int, val_split: float = 0.2, **kwargs) -> None:
        if val_split != 0.:
            xdata_train, xdata_val, ydata_train, ydata_val = train_test_split(
                xdata, ydata, test_size=val_split)

            train_loader = ModelGNN.data_loader(xdata_train, ydata_train)
            val_loader = ModelGNN.data_loader(xdata_val, ydata_val)
        else:
            train_loader = ModelGNN.data_loader(xdata, ydata)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_function = torch.nn.MSELoss()
        best_val_loss = float('inf')

        for epoch in range(1, epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            for batch in train_loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = loss_function(out, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                num_bathces += 1
            
            avg_train_loss = total_loss / num_batches
            if kwargs.get('verbose', True):
                logger.info(f'Epoch {epoch:03d} -- Train Loss {avg_train_loss:.4f}')
            
            if val_split != 0:
                self.model.eval()
                val_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_out = self.model(val_batch.x, val_batch.edge_index, val_batch.batch)
                        val_loss += loss_function(val_out, val_batch.y).item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches
                if kwargs.get('verbose', True):
                    logger.info(f'\t\t -- Val Loss {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if kwargs.get('save_best', True):
                    model_name = kwargs.get('name', 'model.pt')
                    self.save(model_name)
                    logger.info(f'\t\t -- Model saved ({model_name})')

    def predict(self, xdata: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predict_loader = self.data_loader(xdata)
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch in predict_loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                predictions.extend(out.cpu())
        
        predictions = np.array(predictions)

        return predictions[:, 0], predictions[:, 1]

class GNNRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        return self.lin2(x)
