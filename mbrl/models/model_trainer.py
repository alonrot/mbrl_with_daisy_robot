class ModelTrainer:

    def setup(self, model):
        raise NotImplemented("Subclass must implement")

    def train_model(self, training_dataset, testing_dataset, cfg):
        raise NotImplemented("Subclass must implement")