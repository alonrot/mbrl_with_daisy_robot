def default_state_transformer(state):
    return state

class DefaultTargetTransformer:
    @staticmethod
    def forward(s0, s1):
        return s1 - s0

    @staticmethod
    def reverse(s0, predicted_s1):
        return s0 + predicted_s1


class NoOpTargetTransformer:
    @staticmethod
    def forward(_s0, s1):
        return s1

    @staticmethod
    def reverse(_s0, predicted_s1):
        return predicted_s1
