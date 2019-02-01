import tensorflow as tf


class Convolution(object):
    def __init__(self,
                 num_in_channels,
                 num_out_channels,
                 data_format,
                 kernel_size,
                 stride,
                 padding,
                 activation_fn,
                 normalizer_fn,
                 normalizer_params,
                 weights_initializer,
                 weights_regularizer,
                 biases_initializer,
                 biases_regularizer,
                 trainable):

        self._data_format = data_format
        filter_height = kernel_size
        filter_width = kernel_size
        self._strides = [1, 1, 1, 1]  # init N H W C indexes
        self._strides[data_format.index('H')] = stride
        self._strides[data_format.index('W')] = stride
        self._padding = padding
        self._activation_fn = activation_fn
        if normalizer_fn is not None:
            raise NotImplementedError
        if normalizer_params is not None:
            raise NotImplementedError
        self._trainable = trainable

        # create filter
        self._filter = self._create_variable(name_or_scope='filter',
                                             shape=[filter_height, filter_width, num_in_channels, num_out_channels],
                                             initializer=weights_initializer,
                                             regularizer=weights_regularizer)

        # create biases
        if biases_initializer is None:
            self._biases = None
        else:
            biases_shape = [1, 1, 1]
            biases_shape[data_format.index('C')-1] = num_out_channels
            self._biases = self._create_variable(name_or_scope='biases',
                                                 shape=biases_shape,
                                                 initializer=biases_initializer,
                                                 regularizer=biases_regularizer)

    # can be overridden
    def _create_variable(self, name_or_scope, shape, initializer, regularizer):
        return tf.get_variable(name=name_or_scope, shape=shape, initializer=initializer, regularizer=regularizer,
                               trainable=self._trainable)

    def __call__(self, inputs):  # [batch, ih, iw, ic]

        # convolve
        outputs = tf.nn.conv2d(input=inputs,
                               filter=self._filter,
                               strides=self._strides,
                               padding=self._padding,
                               data_format=self._data_format)

        # bias
        if self._biases is not None:
            outputs += self._biases

        # apply activation
        if self._activation_fn is not None:
            outputs = self._activation_fn(outputs)

        return outputs  # [batch, oh, ow, oc]
