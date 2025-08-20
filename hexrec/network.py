"""Module to handle neural networks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
import math
from pathlib import Path
from typing import Tuple

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import keras
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split

from hexrec import HEXREC_MODELS

class ModelBase(ABC):
    """Abstract base class for the definition of a neural network model class.
    Subclasses must implement the core methods for saving and loading models,
    loading pretrained models, training and prediction.

    This class cannot be instantiated directly.
    """

    @abstractmethod
    def save(self, name: str) -> Path:
        """Save the model to the default "models" folder with the given name.

        This method must be implemented by all subclasses. 

        Args:
            name (str): The name of the file to save the model as.

        Returns:
            Path: The path to the saved model file.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ModelBase':
        """
        Load the model from the given path.

        This method must be implemented by all subclasses.

        Args:
            path (str): The path to the model file to load.

        Returns:
            ModelBase: An instance of the loaded model.
        """
        pass

    @classmethod
    @abstractmethod
    def load_pretrained(cls) -> 'ModelBase':
        """
        Load the pre-trained model provided by the package.

        This method must be implemented by all subclasses.

        Returns:
            ModelBase: An instance of the pre-trained model.
        """
        pass

    @abstractmethod
    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int,
            val_split: float = 0.2, **kwargs):
        """
        Train the neural network model.

        This method must be implemented by all subclasses.

        Args:
            xdata (np.ndarray): Input training data.
            ydata (np.ndarray): Target training data.
            epochs (int): Number of training epochs.
            val_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            **kwargs: Additional keyword arguments for training configuration.
        """
        pass

    @abstractmethod
    def predict(self, xdata: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given input data.

        This method must be implemented by all subclasses.

        Args:
            xdata (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted output data.
        """
        pass

@dataclass
class ModelDNN(ModelBase):
    """
    Concrete implementation of a deep dense neural network model.

    This class manages the saving, loading, training and prediction
    of a deep dense neural network according to the ModelBase interface.
    """
    model: keras.models.Model

    def save(self, name: str) -> Path:
        """Save the DNN model to the default "models" folder with the given name.

        Args:
            name (str): The name of the file to save the model as.

        Returns:
            Path: The path to the saved model file.
        """
        output_path = HEXREC_MODELS / f'{name}.keras'
        self.model.save(output_path)

        return output_path

    @classmethod
    def load(cls, path: str) -> 'ModelDNN':
        """
        Load the DNN model from the given path.

        Args:
            path (str): The path to the model file to load.

        Returns:
            ModelDNN: An instance of the ModelDNN with the loaded architecture.
        """
        model = keras.models.load_model(path)

        return cls(model)

    @classmethod
    def load_pretrained(cls) -> 'ModelDNN':
        """
        Load the pre-trained DNN model provided by the package.

        Returns:
            ModelBase: An instance of the pre-trained DNN model.
        """
        with resources.path('hexrec.models', 'modelDNN.keras') as model_path:
            return cls.load(str(model_path))

    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int,
              val_split: float = 0.2, **kwargs):
        """
        Train the DNN model.

        Args:
            xdata (np.ndarray): Input training data.
            ydata (np.ndarray): Target training data.
            epochs (int): Number of training epochs.
            val_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            **kwargs: Additional keyword arguments for training configuration.
        """
        return self.model.fit(xdata, ydata, epochs=epochs,
                       validation_split=val_split, **kwargs)

    def predict(self, xdata: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for the given input data.

        Args:
            xdata (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted output data.
        """
        input_layer_name = self.model.layers[0].name
        prediction = self.model({input_layer_name: xdata})
        prediction = prediction.numpy()

        return prediction

@dataclass
class ModelGNN(ModelBase):
    """
    Concrete implementation of a graph neural network model.

    This class manages the saving, loading, training and prediction
    of a graph neural network according to the ModelBase interface.
    """
    model: torch.nn.Module

    def save(self, name: str) -> Path:
        """Save the GNN model to the default "models" folder with the given name.

        Args:
            name (str): The name of the file to save the model as.

        Returns:
            Path: The path to the saved model file.
        """
        output_path = HEXREC_MODELS / f'{name}.pt'
        torch.save(self.model.state_dict(), output_path)
        return output_path

    def load(self, path: str) -> None:
        """
        Load the GNN model from the given path.

        Args:
            path (str): The path to the model file to load.
        """
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    @classmethod
    def load_pretrained(cls) -> 'ModelGNN':
        """
        Load the pre-trained GNN model provided by the package.

        Returns:
            ModelGNN: An instance of the pre-trained GNN model.
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
    def data_loader(xdata: np.ndarray, ydata: np.ndarray = None, batch_size: int = 16,
                    **kwargs) -> DataLoader:
        """Prepare data for training or prediction

        Args:
            xdata (np.ndarray): data for training or prediction. Shape must be (7, 3) for 
                single data, or (N, 7, 3) for multiple data

            ydata (np.ndarray, optional): data for training. Shape must be (2,) for single data
                or (N, 2) for multiple data

        Returns:
            DataLoader: data prepared for training and prediction
        """
        # Check if there is only an event (7, 3)
        if len(xdata.shape) == 2:
            assert xdata.shape == (7, 3)
            dataset = [ModelGNN.event_to_graph(xdata, ydata)]
        # Or multiple events (N, 7, 3)
        elif len(xdata.shape) == 3:
            assert xdata.shape[1:] == (7, 3)
            if ydata is not None:
                dataset = [ModelGNN.event_to_graph(_xdata, _ydata) for _xdata, _ydata
                           in zip(xdata, ydata)]
            else:
                dataset = [ModelGNN.event_to_graph(_xdata) for _xdata in xdata]
        else:
            raise RuntimeError(f'''Incorrect xdata dimension: {xdata.shape}.
                               Must be (7, 3) or (N, 7, 3)''')

        dataset_loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

        return dataset_loader

    def train(self, xdata: np.ndarray, ydata: np.ndarray, epochs: int, val_split: float = 0.2,
              **kwargs) -> None:
        """
        Train the GNN model.

        Args:
            xdata (np.ndarray): Input training data.
            ydata (np.ndarray): Target training data.
            epochs (int): Number of training epochs.
            val_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            **kwargs: Additional keyword arguments for training configuration.
        """
        if val_split != 0.:
            xdata_train, xdata_val, ydata_train, ydata_val = train_test_split(
                xdata, ydata, test_size=val_split)

            logger.info('Preparing training data')
            train_loader = ModelGNN.data_loader(xdata_train, ydata_train)
            logger.info('Preparing validation data')
            val_loader = ModelGNN.data_loader(xdata_val, ydata_val)
        else:
            logger.info('Preparing training data for training')
            train_loader = ModelGNN.data_loader(xdata, ydata)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_function = torch.nn.MSELoss()
        best_val_loss = float('inf')

        self.history = {'loss':[], 'val_loss':[]}
        for epoch in range(1, epochs+1):
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
                num_batches += 1

            avg_train_loss = total_loss / num_batches
            self.history['loss'].append(avg_train_loss)
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
                self.history['val_loss'].append(avg_val_loss)
                if kwargs.get('verbose', True):
                    logger.info(f'\t\t -- Val Loss {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if kwargs.get('save_best', True):
                    model_name = kwargs.get('name', 'model')
                    self.save(model_name)
                    logger.info(f'\t\t -- Model saved ({model_name})')

        self.plot_history()

    def plot_history(self) -> None:
        """Plot history of loss and val_loss metrics over training epochs
        """
        plt.plot(self.history['loss'], label='loss')
        plt.plot(self.history['val_loss'], label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error')
        plt.legend()
        plt.show()

    def predict(self, xdata: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for the given input data.

        Args:
            xdata (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted output data.
        """
        predict_loader = self.data_loader(xdata)
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch in predict_loader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                predictions.extend(out.cpu())

        predictions = np.array(predictions)

        return predictions

class GNNRegression(torch.nn.Module):
    """Defines the architecture of the default Graph Neural Network (GNN) regression model.

    The model consists of two graph convolutional layers followed by
    two fully connected linear layers. It performs graph-level regression
    by applying global mean pooling after convolutional layers.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        """Perform a forward pass through the network.

        Applies two graph convolutional layers with ReLU activations,
        followed by global mean pooling, then two fully connected layers
        with ReLU activation before the final output layer.
        """
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        return self.lin2(x)
