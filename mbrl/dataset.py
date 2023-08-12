import torch.utils.data

class SAS:
    """
    SAS stands forstates (S), actions (A), and next states (S).
    """
    def __init__(self, s0, a, s1):
        assert torch.is_tensor(s0)
        assert torch.is_tensor(a)
        assert torch.is_tensor(s1)
        assert s0.type
        self.s0 = s0
        self.a = a
        self.s1 = s1

    def to_device(self, device):
        self.s0 = self.s0.to(device=device)
        self.s1 = self.s1.to(device=device)
        self.a = self.a.to(device=device)

    def __repr__(self):
        return "s0={}, a={}, s1={}".format(self.s0, self.a, self.s1)

    def __eq__(self, other):
        if not isinstance(other, SAS):
            return NotImplemented

        return (torch.equal(self.s0, other.s0) and
                torch.equal(self.a, other.a) and
                torch.equal(self.s1, other.s1))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.dataset = [(x[i], y[i]) for i in range(x.shape[0])]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class TensorList(object):
    """
    This is a list container that can store a list of vectors (each having the
    same dimension) into a single huge torch.Tensor. This can reduce the time to
    serialize/deserialize the data significantly comparing to storing them in
    Python's built-in list if there are more than thousands of elements.
    """
    def __init__(self, initial_capacity=10):
        self.data = torch.Tensor()
        self.size = 0
        self.initial_capacity = initial_capacity
        self.shape = None

    def append(self, data):

        # the dim has to be smaller than 2 and can be squeezed into a 1-D vector
        assert data.dim() == 1 or (data.dim() == 2 and (data.squeeze().dim() == 1)), (
            f"The shape of data can only be N, 1xN, or Nx1 not {data.shape}")

        # the data we're trying to append has to have the same shape as before
        assert self.shape is None or self.shape == list(data.shape)

        # if self.data is still an empty Tensor (just initialized)
        if self.data.nelement() == 0:
            self.shape = list(data.shape)

            # Get the dimension of data. Don't use squeeze (it will squash 1-D
            # vector down to a 0-D scalar)
            dim = data.reshape(-1).shape[0]
            self.data = torch.zeros(self.initial_capacity, dim,
                                    dtype=data.dtype, device=data.device)
        else:
            self._expand_if_necessary()

        self.data[self.size] = data.reshape(1, -1)
        self.size += 1

    def _expand_if_necessary(self):
        capacity, dim = self.data.shape

        if self.size < capacity:
            return

        new_data = torch.zeros((capacity * 2, dim), dtype=self.data.dtype,
                               device=self.data.device)
        new_data[:capacity] = self.data
        self.data = new_data

    def __getitem__(self, idx):
        # if idx < 0, wrap around according to self.size not self.data.shape[0]
        index = idx + self.size if idx < 0 else idx
        if not (0 <= index < self.size):
            raise IndexError(f"index {idx} is out of range. size = {self.size}")

        return self.data[index]

    def __repr__(self):
        return f"TensorList with {len(self)} elements of shape {self.shape}"

    def __len__(self):
        return self.size

    def __eq__(self, other):
        if not isinstance(other, TensorList):
            return NotImplemented

        return torch.equal(self.data[:self.size], other.data[:self.size])

class SASDataset(torch.utils.data.Dataset):
    """
    A dataset container storing states (S), actions (A), and next states (S).
    """
    def __init__(self, container=TensorList):
        self.states0 = container()
        self.actions = container()
        self.states1 = container()

    def add(self, sars):
        self.states0.append(sars.s0)
        self.actions.append(sars.a)
        self.states1.append(sars.s1)

    def add_episode(self, episode, device):
        for sars in episode:
            sars.to_device(device)
            self.add(sars)

    def __len__(self):
        return len(self.states0)

    def __getitem__(self, idx):
        s0, a, s1 = self.states0[idx], self.actions[idx], self.states1[idx]
        return SAS(s0, a, s1)

    def __eq__(self, other):
        if not isinstance(other, SASDataset):
            return NotImplemented

        return (self.states0 == other.states0 and
                self.actions == other.actions and
                self.states1 == other.states1)
