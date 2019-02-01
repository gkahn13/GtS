import numpy as np
import tensorflow as tf

from gcg.envs.spaces.base import Space


class Box(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    """

    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            if np.isscalar(low) and np.isscalar(high):
                self.low = np.array([low])
                self.high = np.array([high])
            else:
                assert low.shape == high.shape
                self.low = low
                self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

        self.low = self.low.astype(np.float32)
        self.high = self.high.astype(np.float32)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.low.shape)

    def sample_n(self, n):
        return np.random.uniform(low=self.low, high=self.high, size=(n, *self.low.shape))

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    @property
    def shape(self):
        return self.low.shape

    @property
    def flat_dim(self):
        return np.prod(self.low.shape)

    @property
    def bounds(self):
        return self.low, self.high

    @property
    def volume(self):
        return np.prod(self.high - self.low, dtype=np.float32)

    def flatten(self, x):
        return np.asarray(x).flatten()

    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)

    def flatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))

    def unflatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0],) + self.shape)

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Box) and np.allclose(self.low, other.low) and \
               np.allclose(self.high, other.high)

    def __hash__(self):
        return hash((self.low, self.high))

    def new_tensor_variable(self, name, extra_dims, flatten=True):
        if flatten:
            return tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.flat_dim], name=name)
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + list(self.shape), name=name)

    @property
    def dtype(self):
        return tf.float32
