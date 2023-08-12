from mbrl.models.model_type import ModelType


class AbstractModel(ModelType):
    def forward(self, *x):
        raise NotImplemented("Subclass must implement")

    def get_input_size(self):
        raise NotImplemented("Subclass must implement")

    def get_output_size(self):
        raise NotImplemented("Subclass must implement")
