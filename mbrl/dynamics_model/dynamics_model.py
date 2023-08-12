import torch
from dotmap import DotMap

from mbrl.models import ModelType


class DynamicsModel(ModelType):

    def predict(self, states: torch.TensorType, actions: torch.TensorType) -> torch.TensorType:
        """
        predicts the next state and  reward given a state and an action
        :param states: either a state vector os a matrix containing state vectors
        :param actions: either a action vector os a matrix containing action vectors
        :return: new state vector or state vectors, based on input dimensionality
        """
        raise NotImplementedError("Subclass must implement this function")

    # Train the dynamics model
    def train(self, training_dataset, testing_dataset, training_params):
        train_log = DotMap()
        return train_log

    # Reinitialize underlying model, forgetting any previous training
    def reset(self):
        pass
