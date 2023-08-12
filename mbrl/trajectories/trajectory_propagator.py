from mbrl.optimizers import ActionSequence

class TrajectoryPropagator:
    def compute_trajectories_and_returns(self, model, state0, actions, reward_func) -> ActionSequence:
        raise NotImplementedError("Subclass must implement this function")
