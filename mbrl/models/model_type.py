

class ModelType:
    def get_ensemble_size(self):
        return 1

    def is_ensemble(self):
        return False

    def is_deterministic(self):
        raise NotImplemented("Subclass must implement")

    def is_probabilistic(self):
        return not self.is_deterministic()

