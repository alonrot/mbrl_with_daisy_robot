import logging

import torch

from mbrl import utils
from .dynamics_model import DynamicsModel

log = logging.getLogger(__name__)


class NNBasedDynamicsModel(DynamicsModel):

    def __init__(self,
                 jit,
                 model,
                 trainer,
                 device,
                 state_size,
                 action_size,
                 state_transformer,
                 target_transformer):
        self.device = device
        self.model_cfg = model
        self.trainer_cfg = trainer
        self.jit = jit
        self.state_transformer = utils.get_static_method(state_transformer)
        self.forward_target_transformer = utils.get_static_method(f"{target_transformer}.forward")
        self.reverse_target_transformer = utils.get_static_method(f"{target_transformer}.reverse")
        self.input_size = self.state_transformer(torch.zeros(state_size)).shape[0] + action_size
        self.output_size = state_size
        self.nn = None
        self.trainer = None
        self.network = None
        # jitted individual ensemble models if the model is an ensemble. used in predict_one_model
        self.jitted_ensemble_models = None
        self.reset()

    def jit_model(self):
        if self.jit:
            batch_size = 1000

            # TODO alonrot added:
            # The #original line doesn't work if the model was trained in cuda, and is loaded on cpu
            if torch.cuda.is_available(): # Only if the machine supports cuda there's a decision to be made
                example_input = torch.zeros(batch_size, self.input_size).to(device="cuda") # original
            else:
                example_input = torch.zeros(batch_size, self.input_size)
            # NOTE: even if the machine supports cuda, by default, tensors are allocated in cpu, unless otherwise specified.

            self.network = torch.jit.trace(self.nn, example_input)
            self.network(example_input)  # jit

            if self.nn.is_ensemble():
                models = self.nn.get_models()
                self.jitted_ensemble_models = []
                for model in models:
                    jm = torch.jit.trace(model, example_input)
                    jm(example_input)  # jit
                    self.jitted_ensemble_models.append(jm)
        else:
            self.network = self.nn
            if self.nn.is_ensemble():
                self.jitted_ensemble_models = self.nn.get_models()

    def reset(self):
        self.nn = utils.instantiate(self.model_cfg, self.input_size, self.output_size, self.device)
        self.network = self.nn
        self.jit_model()
        if self.trainer_cfg is not None:
            self.trainer = utils.instantiate(self.trainer_cfg)
            self.trainer.setup(self.nn)

    @staticmethod
    def assert_valid_input(states: torch.TensorType, actions: torch.TensorType):
        assert torch.is_tensor(states)
        assert torch.is_tensor(actions)
        assert states.dim() == 2
        assert actions.dim() == 2
        assert states.size(0) == actions.size(0)

    def prepare_input(self, states: torch.TensorType, actions: torch.TensorType) -> torch.TensorType:
        transformed_states = self.state_transformer(states)
        return torch.cat([transformed_states, actions], dim=1)

    def predict(self, states: torch.TensorType, actions: torch.TensorType) -> torch.TensorType:
        NNBasedDynamicsModel.assert_valid_input(states, actions)
        with torch.no_grad():
            inputs = self.prepare_input(states, actions)
            outputs = self.network.forward(inputs)
        assert not torch.isnan(outputs).any()

        if not self.is_ensemble():
            # Single output
            if self.is_deterministic():
                # D
                outputs = self.reverse_target_transformer(states, outputs)
            elif self.is_probabilistic():
                # P
                outputs[:, :, 0] = self.reverse_target_transformer(states, outputs[:, :, 0])
        else:
            # Ensemble output
            if self.is_deterministic():
                # DE
                for e in range(outputs.size(2)):
                    outputs[:, :, e] = self.reverse_target_transformer(states, outputs[:, :, e])
            elif self.is_probabilistic():
                # PE
                for e in range(outputs.size(2)):
                    outputs[:, :, e, 0] = self.reverse_target_transformer(states, outputs[:, :, e, 0])

        return outputs

    def predict_one_model(self, states: torch.TensorType, actions: torch.TensorType, index: int) -> torch.TensorType:
        """
        Predicts the output of a single model in the ensemble on a given state and action batch
        :param states: state batch
        :param actions:  action batch
        :param index: index of model in ensemble
        :return:
        """
        # only makes sense for underlying ensemble model
        NNBasedDynamicsModel.assert_valid_input(states, actions)
        assert self.is_ensemble()
        assert index >= 0
        assert index < self.nn.get_ensemble_size()
        with torch.no_grad():
            inputs = self.prepare_input(states, actions)
            outputs = self.jitted_ensemble_models[index].forward(inputs)

        # Single output
        if self.is_deterministic():
            outputs = self.reverse_target_transformer(states, outputs)
        elif self.is_probabilistic():
            outputs[:, :, 0] = self.reverse_target_transformer(states, outputs[:, :, 0])
        return outputs

    def train(self, training_dataset, testing_dataset, training_params):

        training_ds = utils.convert_to_dataset(training_dataset,
                                               target_transformer=self.forward_target_transformer,
                                               state_transformer=self.state_transformer)
        testing_ds = utils.convert_to_dataset(testing_dataset,
                                              target_transformer=self.forward_target_transformer,
                                              state_transformer=self.state_transformer)
        return self.trainer.train_model(self.nn, training_ds, testing_ds, training_params)

    def get_ensemble_size(self):
        return self.nn.get_ensemble_size()

    def is_ensemble(self):
        return self.nn.is_ensemble()

    def is_deterministic(self):
        return self.nn.is_deterministic()

    def is_probabilistic(self):
        return self.nn.is_probabilistic()

    def change_internal_device_to(self,device_desired):
        """
        
        author: alonrot

        This function changes the device of the entire class. This is useful if we train the dynamics model on a GPU, 
        but later we want to load this entire class on a computer without GPU.
        The model should be changed either before saving it ir right after loading it.

        .. note::
            Has only been tested with "probabilistic ensemble" (pe). Needs to be tested with p, de, and d
        
        :param device: "cpu" or "cuda"
        :return: None
        """
        assert device_desired == "cpu" or device_desired == "cuda", "device_desired must be {cpu,cuda}"

        if self.is_ensemble():
            for k in range(self.get_ensemble_size()):
                # self.nn.models[k].to(device_curr) # Has no effect
                self.nn.models[k].device = device_desired
        
        self.nn.device = device_desired

        return

    # do not save jitted neural net
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['network']
        del state['jitted_ensemble_models']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.jit_model()

    def set_jit(self, jit):
        self.jit = jit
        self.jit_model()
