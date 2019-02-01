import tensorflow as tf


def convnn(inputs,
           conv_class,
           conv_args,
           filters,
           kernels,
           strides,
           padding,
           hidden_activation,
           output_activation,
           trainable=True,
           normalizer_fn=None,
           normalizer_params=None,
           scope='convnn',
           dtype=tf.float32,
           data_format='NCHW',
           reuse=False,
           is_training=True,
           global_step_tensor=None):

    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(len(kernels)):
            activation = hidden_activation if i < len(kernels) - 1 else output_activation

            with tf.variable_scope("conv{}".format(i)):
                conv_call_fn = conv_class(
                    num_in_channels=next_layer_input.get_shape()[data_format.index('C')].value,
                    num_out_channels=filters[i],
                    data_format=data_format,
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=padding,
                    activation_fn=activation,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                    biases_initializer=tf.constant_initializer(0., dtype=dtype),
                    biases_regularizer=None,
                    trainable=trainable,
                    **conv_args)

                next_layer_input = conv_call_fn(next_layer_input)

    output = next_layer_input
    return output


def fullyconnectednn(inputs,
                     fullyconnected_class,
                     fullyconnected_args,
                     hidden_layers,
                     output_dim,
                     hidden_activation,
                     output_activation,
                     trainable=True,
                     normalizer_fn=None,
                     normalizer_params=None,
                     scope='fcnn',
                     reuse=False,
                     T=None,
                     is_training=True,
                     global_step_tensor=None):
    if T is None:
        assert(len(inputs.get_shape()) == 2)
        next_layer_input = inputs
    else:
        assert(len(inputs.get_shape()) == 3)
        assert(inputs.get_shape()[1].value == T)
        next_layer_input = tf.reshape(inputs, (-1, inputs.get_shape()[-1].value))

    with tf.variable_scope(scope, reuse=reuse):
        dims = hidden_layers + [output_dim]
        for i, dim in enumerate(dims):
            with tf.variable_scope('l{0}'.format(i)):
                activation = hidden_activation if i < len(dims) - 1 else output_activation

                fc_call_fn = fullyconnected_class(
                    num_inputs=next_layer_input.get_shape()[1].value,
                    num_outputs=dim,
                    activation_fn=activation,
                    trainable=trainable,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    **fullyconnected_args)

                next_layer_input = fc_call_fn(next_layer_input)

        output = next_layer_input

        if T is not None:
            output = tf.reshape(output, (-1, T, output.get_shape()[-1].value))

    return output


def rnn(inputs,
        rnncell_class,
        rnncell_args,
        num_cells,
        state_tuple_size,
        initial_state=None,
        num_units=None,
        trainable=True,
        dtype=tf.float32,
        scope='rnn',
        reuse=False,
        is_training=True):
    """
    inputs is shape [batch_size x T x features].
    """
    if initial_state is not None:
        assert (num_units is None)

        if state_tuple_size == 1:
            initial_state = tf.split(initial_state, num_cells, axis=1)
            num_units = initial_state[0].get_shape()[1].value
        else:
            states = tf.split(initial_state, 2 * num_cells, axis=1)
            num_units = states[0].get_shape()[1].value
            initial_state = []
            for i in range(num_cells):
                initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(states[i * 2], states[i * 2 + 1]))

        initial_state = tuple(initial_state)
    else:
        assert (num_units is not None)

    cells = []
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(num_cells):
            num_inputs = inputs.get_shape()[-1].value if i == 0 else num_units

            cell = rnncell_class(
                num_units,
                dtype=dtype,
                num_inputs=num_inputs,
                weights_scope='{0}_{1}'.format(type(rnncell_class).__name__, i),
                trainable=trainable,
                **rnncell_args)

            cells.append(cell)

        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(
            multi_cell,
            tf.cast(inputs, dtype),
            initial_state=initial_state,
            dtype=dtype,
            time_major=False)

    return outputs
