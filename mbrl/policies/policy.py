from mbrl.optimizers import ActionSequence


class Policy:
    def plan_action_sequence(self, state) -> ActionSequence:
        raise NotImplementedError("Subclass must implement this function")

    def reset_trajectory(self):
        """
        Reset any state associated with the currently computed trajectory
        """
        pass


    def setup(self, *args):
        """
        Configure policy
        :param args: variable arguments list, specific subclasses may have different arguments here
        and caller is expected to pass the correct ones.
        """
        pass
