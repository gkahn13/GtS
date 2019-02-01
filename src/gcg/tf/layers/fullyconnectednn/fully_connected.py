import tensorflow as tf


class FullyConnected(object):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 activation_fn=None,
                 normalizer_fn=None,
                 normalizer_params=None,
                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                 weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                 biases_initializer=tf.constant_initializer(0.),
                 biases_regularizer=None,
                 trainable=True):

        self._activation_fn = activation_fn
        self._normalizer_fn = normalizer_fn
        self._normalizer_params = normalizer_params
        self._trainable = trainable

        # create weight
        self._weights = self._create_variable("weights", [num_inputs, num_outputs], weights_initializer, weights_regularizer)

        # create biases
        if biases_initializer is None:
            self._biases = None
        else:
            self._biases = self._create_variable("biases", [num_outputs], biases_initializer, biases_regularizer)

    # can be overridden
    def _create_variable(self, name_or_scope, shape, initializer, regularizer):
        return tf.get_variable(name=name_or_scope, shape=shape, initializer=initializer, regularizer=regularizer,
                               trainable=self._trainable)

    def __call__(self, inputs):
        output = tf.matmul(inputs, self._weights)

        if self._normalizer_fn is not None:
            output = self._normalizer_fn(output, **self._normalizer_params)

        if self._biases is not None:
            output += self._biases

        if self._activation_fn is not None:
            output = self._activation_fn(output)

        return output
