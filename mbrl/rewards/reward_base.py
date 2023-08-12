from abc import ABC,abstractmethod # Abstract Base Classes | See https://docs.python.org/3/library/abc.html

class Reward(ABC):
    """
    Abstract class, to be inherited from other task-specific reward classes
    """

    # A class that has a metaclass derived from ABCMeta cannot be instantiated unless all of its abstract methods and properties are overridden.
    # The abstract methods can be called using any of the normal ‘super’ call mechanisms.

    @abstractmethod
    def __init__(self):
        """
        Define here the task goal
        """
        raise NotImplementedError("Implement in child class")

    @abstractmethod
    def get_reward_signal(self,state_curr,action_curr):
        """
        Define here the reward function
        """
        raise NotImplementedError("Implement in child class")

    def update_goal(self):
        """
        Optional method to update the goal
        """
        raise NotImplementedError("Implement in child class")