import math

import numpy as np


class Function:

    def __init__(self, input_size, output_size, output_names):
        self.input_sz = input_size
        self.output_sz = output_size
        self.output_names = output_names

    def output_dim_name(self, i):
        return self.output_names[i]

    def output_size(self):
        return self.output_sz

    def input_size(self):
        return self.input_sz

    def __call__(self, input1):
        raise NotImplemented("subclass must implement")

    def __repr__(self):
        return ','.join(self.output_names)

    def lower_bound(self):
        return np.full(self.output_size(), -10)

    def upper_bound(self):
        return np.full(self.output_size(), 10)


class Sin1D(Function):
    def __init__(self):
        super().__init__(1, 1, ['sin(x)'])

    def __call__(self, input1):
        return np.array([math.sin(input1)])


class SinCos2D(Function):
    def __init__(self):
        super().__init__(1, 2, ['sin(x)', 'cos(x)'])

    def __call__(self, input1):
        return np.array([math.sin(input1), math.cos(input1)])


class Constant(Function):
    def __init__(self, *args):
        super().__init__(1, 1, [str(c) for c in args])
        c = np.array(args)
        assert isinstance(c, np.ndarray)
        self.constant = c

    def __call__(self, _):
        return self.constant


# https://en.wikipedia.org/wiki/Rosenbrock_function
class Banana(Function):
    def __init__(self, a, b):
        super().__init__(2, 2, [f'banana(a={a}, b={b})'])
        self.a = a
        self.b = b

    def __call__(self, input1):
        x = input1[0]
        y = input1[1]
        aa = self.a - x
        bb = y - x * x
        return np.array([aa * aa + self.b * (bb * bb)])


class TorchBanana(Function):
    def __init__(self, a, b):
        super().__init__(2, 2, [f'banana(a={a}, b={b})'])
        self.a = a
        self.b = b

    def __call__(self, input1):
        x = input1[0]
        y = input1[1]
        aa = self.a - x
        bb = y - x * x
        return aa * aa + self.b * (bb * bb)
