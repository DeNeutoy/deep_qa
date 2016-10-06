import copy
import warnings
from collections import OrderedDict
from typing import Dict, Any

from overrides import overrides

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import InputSpec
from keras.layers import Layer, Recurrent, LSTM, Convolution1D, MaxPooling1D, merge, Dense
from keras.regularizers import l1l2
import theano.tensor.extra_ops as T
import tensorflow as tf
import numpy as np

from ..data.constants import SHIFT_OP, REDUCE2_OP, REDUCE3_OP

class TreeCompositionLSTM(Recurrent):
    '''
    Conceptual differences from LSTM:
    1. Tree LSTM does not differentiate between x and h, because
        tree composition is not applied at every time step (it is applied when the input symbol is a
        reduce) and when it is applied, there is no "current input".
    2. Input sequences are not the
        ones being composed, they are operations on the buffer containing elements corresponding to
        tokens. There isn't one token per timestep like LSTMs.
    3. Single vectors h and c are replaced
        by a stack and buffer of h and c corresponding to the structure processed so far.
    4. Gates are
        applied on two or three elements at a time depending on the type of reduce. Accordingly there
        are two classes of gates: G_2 (two elements) and G_3 (three elements)
    5. G_2 has two forget
        gates, for each element that can be forgotten and G_3 has three.
    '''
    # pylint: disable=invalid-name
    def __init__(self, **kwargs):
        assert "stack_limit" in kwargs, "Specify stack_limit"
        assert "buffer_ops_limit" in kwargs, "Specify buffer_ops_limit"
        assert "output_dim" in kwargs, "Specify output_dim"
        self.stack_limit = kwargs["stack_limit"]
        # buffer_ops_limit is the max of buffer_limit and num_ops. This needs to be one value since
        # the initial buffer state and the list of operations need to be concatenated before passing
        # them to TreeCompositionLSTM
        self.buffer_ops_limit = kwargs["buffer_ops_limit"]
        self.output_dim = kwargs["output_dim"]
        init = kwargs.get("init", "glorot_uniform")
        self.init = initializations.get(init)
        inner_init = kwargs.get("inner_init", "orthogonal")
        self.inner_init = initializations.get(inner_init)
        forget_bias_init = kwargs.get("forget_bias_init", "one")
        self.forget_bias_init = initializations.get(forget_bias_init)
        activation = kwargs.get("activation", "tanh")
        self.activation = activations.get(activation)
        inner_activation = kwargs.get("inner_activation", "hard_sigmoid")
        self.inner_activation = activations.get(inner_activation)
        W_regularizer = kwargs.get("W_regularizer", None)
        U_regularizer = kwargs.get("U_regularizer", None)
        V_regularizer = kwargs.get("V_regularizer", None)
        b_regularizer = kwargs.get("b_regularizer", None)
        # Make two deep copies each of W, U and b since regularizers.get() method modifes them!
        W2_regularizer = copy.deepcopy(W_regularizer)
        W3_regularizer = copy.deepcopy(W_regularizer)
        U2_regularizer = copy.deepcopy(U_regularizer)
        U3_regularizer = copy.deepcopy(U_regularizer)
        b2_regularizer = copy.deepcopy(b_regularizer)
        b3_regularizer = copy.deepcopy(b_regularizer)
        # W, U and b get two copies of each corresponding regularizer
        self.W_regularizers = [regularizers.get(W2_regularizer), regularizers.get(W3_regularizer)] \
                if W_regularizer else None
        self.U_regularizers = [regularizers.get(U2_regularizer), regularizers.get(U3_regularizer)] \
                if U_regularizer else None
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizers = [regularizers.get(b2_regularizer), regularizers.get(b3_regularizer)] \
                if b_regularizer else None
        # TODO(pradeep): Ensure output_dim = input_dim - 1

        self.dropout_W = kwargs["dropout_W"] if "dropout_W" in kwargs else 0.
        self.dropout_U = kwargs["dropout_U"] if "dropout_U" in kwargs else 0.
        self.dropout_V = kwargs["dropout_V"] if "dropout_V" in kwargs else 0.
        if self.dropout_W:
            self.uses_learning_phase = True
        # Pass any remaining arguments of the constructor to the super class' constructor
        super(TreeCompositionLSTM, self).__init__(**kwargs)
        if self.stateful:
            warnings.warn("Current implementation cannot be stateful. \
                    Ignoring stateful=True", RuntimeWarning)
            self.stateful = False
        if self.return_sequences:
            warnings.warn("Current implementation cannot return sequences.\
                    Ignoring return_sequences=True", RuntimeWarning)
            self.return_sequences = False

    def get_initial_states(self, x):
        # The initial buffer is sent into the TreeLSTM as a part of the input.
        # i.e., x is a concatenation of the transitions and the initial buffer.
        # (batch_size, buffer_limit, output_dim+1)
        # We will now separate the buffer and the transitions and initialize the
        # buffer state of the TreeLSTM with the initial buffer value.
        # The rest of the input is the transitions, which we do not need now.

        # Take the buffer out.
        init_h_for_buffer = x[:, :, 1:]  # (batch_size, buffer_limit, output_dim)
        # Initializing all c as zeros.
        init_c_for_buffer = K.zeros_like(init_h_for_buffer)

        # Each element in the buffer is a concatenation of h and c for the corresponding
        # node
        init_buffer = K.concatenate([init_h_for_buffer, init_c_for_buffer], axis=-1)
        # We need a symbolic all zero tensor of size (samples, stack_limit, 2*output_dim) for
        # init_stack The problem is the first dim (samples) is a place holder and not an actual
        # value. So we'll use the following trick
        temp_state = K.zeros_like(x)  # (samples, buffer_ops_limit, input_dim)
        temp_state = K.tile(K.sum(temp_state, axis=(1, 2)),
                            (self.stack_limit, 2*self.output_dim, 1))  # (stack_limit, 2*output_dim, samples)
        init_stack = K.permute_dimensions(temp_state, (2, 0, 1))  # (samples, stack_limit, 2*output_dim)
        return [init_buffer, init_stack]

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        # Defining two classes of parameters:
        # 1) predicate, one argument composition (*2_*)
        # 2) predicate, two arguments composition (*3_*)
        #
        # The naming scheme is an extension of the one used
        # in the LSTM code of Keras. W is a weight and b is a bias
        # *_i: input gate parameters
        # *_fp: predicate forget gate parameters
        # *_fa: argument forget gate parameters (one-arg only)
        # *_fa1: argument-1 forget gate parameters (two-arg only)
        # *_fa2: argument-2 forget gate parameters (two-arg only)
        # *_u: update gate parameters
        # *_o: output gate parameters
        #
        # Predicate, argument composition:
        # W2_i, W2_fp, W2_fa, W2_o, W2_u
        # U2_i, U2_fp, U2_fa, U2_o, U2_u
        # b2_i, b2_fp, b2_fa, b2_o, b2_u
        #
        # Predicate, two argument composition:
        # W3_i, W3_fp, W3_fa1, W3_fa2, W3_o, W3_u
        # U3_i, U3_fp, U3_fa1, U3_fa2, U3_o, U3_u
        # V3_i, V3_fp, V3_fa1, V3_fa2, V3_o, V3_u
        # b3_i, b3_fp, b3_fa1, b3_fa2, b3_o, b3_u

        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]
        # initial states: buffer and stack. buffer has shape (samples, buff_limit, output_dim);
        # stack has shape (samples, stack_limit, 2*output_dim)
        self.states = [None, None]

        # The first dims in all weight matrices are k * output_dim because of the recursive nature
        # of treeLSTM
        if self.consume_less == 'gpu':
            # Input dimensionality for all W2s is output_dim, and there are 5 W2s: i, fp, fa, u, o
            self.W2 = self.init((self.output_dim, 5 * self.output_dim), name='{}_W2'.format(self.name))
            # Input dimensionality for all U2s is output_dim, and there are 5 U2s: i, fp, fa, u, o
            self.U2 = self.init((self.output_dim, 5 * self.output_dim), name='{}_U2'.format(self.name))

            # Input dimensionality for all W3s is output_dim, and there are 6 W2s: i, fp, fa1, fa2, u, o
            self.W3 = self.init((self.output_dim, 6 * self.output_dim), name='{}_W3'.format(self.name))
            # Input dimensionality for all U3s is output_dim, and there are 6 U3s: i, fp, fa1, fa2, u, o
            self.U3 = self.init((self.output_dim, 6 * self.output_dim), name='{}_U3'.format(self.name))
            # Input dimensionality for all V3s is output_dim, and there are 6 V3s: i, fp, fa1, fa2, u, o
            self.V3 = self.init((self.output_dim, 6 * self.output_dim), name='{}_V3'.format(self.name))

            self.b2 = K.variable(np.hstack((np.zeros(self.output_dim),
                                            K.get_value(self.forget_bias_init(self.output_dim)),
                                            K.get_value(self.forget_bias_init(self.output_dim)),
                                            np.zeros(self.output_dim),
                                            np.zeros(self.output_dim))),
                                 name='{}_b2'.format(self.name))
            self.b3 = K.variable(np.hstack((np.zeros(self.output_dim),
                                            K.get_value(self.forget_bias_init(self.output_dim)),
                                            K.get_value(self.forget_bias_init(self.output_dim)),
                                            K.get_value(self.forget_bias_init(self.output_dim)),
                                            np.zeros(self.output_dim),
                                            np.zeros(self.output_dim))),
                                 name='{}_b3'.format(self.name))
            self.trainable_weights = [self.W2, self.U2, self.W3, self.U3, self.V3, self.b2, self.b3]
        else:
            self.W2_i = self.init((self.output_dim, self.output_dim), name='{}_W2_i'.format(self.name))
            self.U2_i = self.init((self.output_dim, self.output_dim), name='{}_U2_i'.format(self.name))
            self.W3_i = self.init((self.output_dim, self.output_dim), name='{}_W3_i'.format(self.name))
            self.U3_i = self.init((self.output_dim, self.output_dim), name='{}_U3_i'.format(self.name))
            self.V3_i = self.init((self.output_dim, self.output_dim), name='{}_V3_i'.format(self.name))
            self.b2_i = K.zeros((self.output_dim,), name='{}_b2_i'.format(self.name))
            self.b3_i = K.zeros((self.output_dim,), name='{}_b3_i'.format(self.name))

            self.W2_fp = self.init((self.output_dim, self.output_dim), name='{}_W2_fp'.format(self.name))
            self.U2_fp = self.init((self.output_dim, self.output_dim), name='{}_U2_fp'.format(self.name))
            self.W2_fa = self.init((self.output_dim, self.output_dim), name='{}_W2_fa'.format(self.name))
            self.U2_fa = self.init((self.output_dim, self.output_dim), name='{}_U2_fa'.format(self.name))
            self.W3_fp = self.init((self.output_dim, self.output_dim), name='{}_W3_fp'.format(self.name))
            self.U3_fp = self.init((self.output_dim, self.output_dim), name='{}_U3_fp'.format(self.name))
            self.V3_fp = self.init((self.output_dim, self.output_dim), name='{}_V3_fp'.format(self.name))
            self.W3_fa1 = self.init((self.output_dim, self.output_dim), name='{}_W3_fa1'.format(self.name))
            self.U3_fa1 = self.init((self.output_dim, self.output_dim), name='{}_U3_fa1'.format(self.name))
            self.V3_fa1 = self.init((self.output_dim, self.output_dim), name='{}_V3_fa1'.format(self.name))
            self.W3_fa2 = self.init((self.output_dim, self.output_dim), name='{}_W3_fa2'.format(self.name))
            self.U3_fa2 = self.init((self.output_dim, self.output_dim), name='{}_U3_fa2'.format(self.name))
            self.V3_fa2 = self.init((self.output_dim, self.output_dim), name='{}_V3_fa2'.format(self.name))
            self.b2_fp = self.forget_bias_init((self.output_dim,), name='{}_b2_fp'.format(self.name))
            self.b2_fa = self.forget_bias_init((self.output_dim,), name='{}_b2_fa'.format(self.name))
            self.b3_fp = self.forget_bias_init((self.output_dim,), name='{}_b3_fp'.format(self.name))
            self.b3_fa1 = self.forget_bias_init((self.output_dim,), name='{}_b3_fa1'.format(self.name))
            self.b3_fa2 = self.forget_bias_init((self.output_dim,), name='{}_b3_fa2'.format(self.name))

            self.W2_u = self.init((self.output_dim, self.output_dim), name='{}_W2_u'.format(self.name))
            self.U2_u = self.init((self.output_dim, self.output_dim), name='{}_U2_u'.format(self.name))
            self.W3_u = self.init((self.output_dim, self.output_dim), name='{}_W3_u'.format(self.name))
            self.U3_u = self.init((self.output_dim, self.output_dim), name='{}_U3_u'.format(self.name))
            self.V3_u = self.init((self.output_dim, self.output_dim), name='{}_V3_u'.format(self.name))
            self.b2_u = K.zeros((self.output_dim,), name='{}_b2_u'.format(self.name))
            self.b3_u = K.zeros((self.output_dim,), name='{}_b3_u'.format(self.name))

            self.W2_o = self.init((self.output_dim, self.output_dim), name='{}_W2_o'.format(self.name))
            self.U2_o = self.init((self.output_dim, self.output_dim), name='{}_U2_o'.format(self.name))
            self.W3_o = self.init((self.output_dim, self.output_dim), name='{}_W3_o'.format(self.name))
            self.U3_o = self.init((self.output_dim, self.output_dim), name='{}_U3_o'.format(self.name))
            self.V3_o = self.init((self.output_dim, self.output_dim), name='{}_V3_o'.format(self.name))
            self.b2_o = K.zeros((self.output_dim,), name='{}_b2_o'.format(self.name))
            self.b3_o = K.zeros((self.output_dim,), name='{}_b3_o'.format(self.name))

            self.trainable_weights = [self.W2_i, self.U2_i, self.W3_i, self.U3_i, self.V3_i, self.b2_i, self.b3_i,
                                      self.W2_fp, self.U2_fp, self.W2_fa, self.U2_fa, self.b2_fp, self.b2_fa,
                                      self.W3_fp, self.U3_fp, self.V3_fp, self.W3_fa1, self.U3_fa1, self.V3_fa1,
                                      self.W3_fa2, self.U3_fa2, self.V3_fa2, self.b3_fp, self.b3_fa1, self.b3_fa2,
                                      self.W2_u, self.U2_u, self.W3_u, self.U3_u, self.V3_u, self.b2_u, self.b3_u,
                                      self.W2_o, self.U2_o, self.W3_o, self.U3_o, self.V3_o, self.b2_o, self.b3_o]

            self.W2 = K.concatenate([self.W2_i, self.W2_fp, self.W2_fa, self.W2_u, self.W2_o])
            self.U2 = K.concatenate([self.U2_i, self.U2_fp, self.U2_fa, self.U2_u, self.U2_o])
            self.W3 = K.concatenate([self.W3_i, self.W3_fp, self.W3_fa1, self.W3_fa2, self.W3_u, self.W3_o])
            self.U3 = K.concatenate([self.U3_i, self.U3_fp, self.U3_fa1, self.U3_fa2, self.U3_u, self.U3_o])
            self.V3 = K.concatenate([self.V3_i, self.V3_fp, self.V3_fa1, self.V3_fa2, self.V3_u, self.V3_o])
            self.b2 = K.concatenate([self.b2_i, self.b2_fp, self.b2_fa, self.b2_u, self.b2_o])
            self.b3 = K.concatenate([self.b3_i, self.b3_fp, self.b3_fa1, self.b3_fa2, self.b3_u, self.b3_o])

        self.regularizers = []
        if self.W_regularizers:
            self.W_regularizers[0].set_param(self.W2)
            self.W_regularizers[1].set_param(self.W3)
            self.regularizers.extend(self.W_regularizers)
        if self.U_regularizers:
            self.U_regularizers[0].set_param(self.U2)
            self.U_regularizers[1].set_param(self.U3)
            self.regularizers.extend(self.U_regularizers)
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V3)
            self.regularizers.append(self.V_regularizer)
        if self.b_regularizers:
            self.b_regularizers[0].set_param(self.b2)
            self.b_regularizers[1].set_param(self.b3)
            self.regularizers.extend(self.b_regularizers)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _one_arg_compose(self, pred_arg):
        # pred_arg: Tensors of size (batch_size, 2, dim) where
        # pred_arg[:,0,:] are arg vectors (h,c) of all samples
        # pred_arg[:,1,:] are pred vectors (h,c) of all samples
        pred_h = pred_arg[:, 1, :self.output_dim]
        pred_c = pred_arg[:, 1, self.output_dim:]
        arg_h = pred_arg[:, 0, :self.output_dim]
        arg_c = pred_arg[:, 0, self.output_dim:]
        if self.consume_less == 'gpu':
            # To optimize for GPU, we would want to make fewer
            # matrix multiplications, but with bigger matrices.
            # So we compute outputs of all gates simultaneously
            # using the concatenated operators W2, U2 abd b2
            z_all_gates = K.dot(pred_h, self.W2) + K.dot(arg_h, self.U2) + self.b2  # (batch_size, 5*output_dim)

            # Now picking the appropriate parts for each gate.
            # All five zs are of shape (batch_size, output_dim)
            z_i = z_all_gates[:, :self.output_dim]
            z_fp = z_all_gates[:, self.output_dim: 2*self.output_dim]
            z_fa = z_all_gates[:, 2*self.output_dim: 3*self.output_dim]
            z_u = z_all_gates[:, 3*self.output_dim: 4*self.output_dim]
            z_o = z_all_gates[:, 4*self.output_dim: 5*self.output_dim]

        else:
            # We are optimizing for memory. Smaller matrices, and
            # more computations. So we use the non-concatenated
            # operators W2_i, U2_i, ..
            z_i = K.dot(pred_h, self.W2_i) + K.dot(arg_h, self.U2_i) + self.b2_i
            z_fp = K.dot(pred_h, self.W2_fp) + K.dot(arg_h, self.U2_fp) + self.b2_fp
            z_fa = K.dot(pred_h, self.W2_fa) + K.dot(arg_h, self.U2_fa) + self.b2_fa
            z_u = K.dot(pred_h, self.W2_u) + K.dot(arg_h, self.U2_u) + self.b2_u
            z_o = K.dot(pred_h, self.W2_o) + K.dot(arg_h, self.U2_o) + self.b2_o

        # Applying non-linearity to get outputs of each gate
        i = self.inner_activation(z_i)
        fp = self.inner_activation(z_fp)
        fa = self.inner_activation(z_fa)
        u = self.inner_activation(z_u)
        c = (i * u) + (fp * pred_c) + (fa * arg_c)
        o = self.inner_activation(z_o)

        # Calculate the composition output. SPINN does not have a non-linearity in the
        # following computation, but the original LSTM does.
        h = o * self.activation(c)

        # Finally return the composed representation for the stack, adding a time
        # dimension and make number of dimensions same as the input
        # to this function
        return K.expand_dims(K.concatenate([h, c]), 1)

    def _two_arg_compose(self, pred_args):
        # pred_args: Matrix of size (samples, 3, dim) where
        # pred_args[:,0,:] are arg2 vectors (h,c) of all samples
        # pred_args[:,1,:] are arg1 vectors (h,c) of all samples
        # pred_args[:,2,:] are pred vectors (h,c) of all samples

        # This function is analogous to _one_arg_compose, except that it operates on
        # two args instead of one. Accordingly, the operators are W3, U3, V3 and b3
        # instead of W2, U2 and b2
        pred_h = pred_args[:, 2, :self.output_dim]
        pred_c = pred_args[:, 2, self.output_dim:]
        arg1_h = pred_args[:, 1, :self.output_dim]
        arg1_c = pred_args[:, 1, self.output_dim:]
        arg2_h = pred_args[:, 0, :self.output_dim]
        arg2_c = pred_args[:, 0, self.output_dim:]
        if self.consume_less == 'gpu':
            z_all_gates = K.dot(pred_h, self.W3) + K.dot(arg1_h, self.U3) + \
                    K.dot(arg2_h, self.V3) + self.b3  # (batch_size, 6*output_dim)

            z_i = z_all_gates[:, :self.output_dim]
            z_fp = z_all_gates[:, self.output_dim: 2*self.output_dim]
            z_fa1 = z_all_gates[:, 2*self.output_dim: 3*self.output_dim]
            z_fa2 = z_all_gates[:, 3*self.output_dim: 4*self.output_dim]
            z_u = z_all_gates[:, 4*self.output_dim: 5*self.output_dim]
            z_o = z_all_gates[:, 5*self.output_dim: 6*self.output_dim]

        else:
            z_i = K.dot(pred_h, self.W3_i) + K.dot(arg1_h, self.U3_i) + \
                    K.dot(arg2_h, self.V3_i) + self.b3_i
            z_fp = K.dot(pred_h, self.W3_fp) + K.dot(arg1_h, self.U3_fp) + \
                    K.dot(arg2_h, self.V3_fp) + self.b3_fp
            z_fa1 = K.dot(pred_h, self.W3_fa1) + K.dot(arg1_h, self.U3_fa1) + \
                    K.dot(arg2_h, self.V3_fa1) + self.b3_fa1
            z_fa2 = K.dot(pred_h, self.W3_fa2) + K.dot(arg1_h, self.U3_fa2) + \
                    K.dot(arg2_h, self.V3_fa2) + self.b3_fa2
            z_u = K.dot(pred_h, self.W3_u) + K.dot(arg1_h, self.U3_u) + \
                    K.dot(arg2_h, self.V3_u) + self.b3_u
            z_o = K.dot(pred_h, self.W3_o) + K.dot(arg1_h, self.U3_o) + \
                    K.dot(arg2_h, self.V3_o) + self.b3_o

        i = self.inner_activation(z_i)
        fp = self.inner_activation(z_fp)
        fa1 = self.inner_activation(z_fa1)
        fa2 = self.inner_activation(z_fa2)
        u = self.inner_activation(z_u)
        c = (i * u) + (fp * pred_c) + (fa1 * arg1_c) + (fa2 * arg2_c)
        o = self.inner_activation(z_o)

        h = o * self.activation(c)

        return K.expand_dims(K.concatenate([h, c]), 1)

    def step(self, x, states):
        # This function is called at each timestep. Before calling this, Keras' rnn
        # dimshuffles the input to have time as the leading dimension, and iterates over
        # it So,
        # x: (samples, input_dim) = (samples, x_op + <current timestep's buffer input>)
        #
        # We do not need the current timestep's buffer input here, since the buffer
        # state is the one that's relevant. We just want the current op from x.

        buff = states[0] # Current state of buffer
        stack = states[1] # Current state of stack

        step_ops = x[:, 0] #(samples, 1), current ops for all samples.

        # We need to make tensors from the ops vectors, one to apply to all dimensions
        # of stack, and the other for the buffer.
        # For the stack:
        # Note stack's dimensionality is 2*output_dim because it holds both h and c
        stack_tiled_step_ops = K.permute_dimensions(
                K.tile(step_ops, (self.stack_limit, 2 * self.output_dim, 1)),
                (2, 0, 1))  # (samples, stack_limit, 2*dim)

        # For the buffer:
        buff_tiled_step_ops = K.permute_dimensions(
                K.tile(step_ops, (self.buffer_ops_limit, 2 * self.output_dim, 1)),
                (2, 0, 1))  # (samples, buff_len, 2*dim)

        shifted_stack = K.concatenate([buff[:, :1], stack], axis=1)[:, :self.stack_limit]
        one_reduced_stack = K.concatenate([self._one_arg_compose(stack[:, :2]),
                                           stack[:, 2:],
                                           K.zeros_like(stack)[:, :1]],
                                          axis=1)
        two_reduced_stack = K.concatenate([self._two_arg_compose(stack[:, :3]),
                                           stack[:, 3:],
                                           K.zeros_like(stack)[:, :2]],
                                          axis=1)
        shifted_buff = K.concatenate([buff[:, 1:], K.zeros_like(buff)[:, :1]], axis=1)

        stack = K.switch(K.equal(stack_tiled_step_ops, SHIFT_OP), shifted_stack, stack)
        stack = K.switch(K.equal(stack_tiled_step_ops, REDUCE2_OP), one_reduced_stack, stack)
        stack = K.switch(K.equal(stack_tiled_step_ops, REDUCE3_OP), two_reduced_stack, stack)
        buff = K.switch(K.equal(buff_tiled_step_ops, SHIFT_OP), shifted_buff, buff)

        stack_top_h = stack[:, 0, :self.output_dim] # first half of the top element for all samples

        return stack_top_h, [buff, stack]

    def get_constants(self, x):
        # TODO(pradeep): The function in the LSTM implementation produces dropout multipliers
        # to apply on the input if dropout is applied on the weights W and U. Ignoring
        # dropout for now.
        constants = []
        if 0 < self.dropout_W < 1 or 0 < self.dropout_U < 1 or 0 < self.dropout_V < 1:
            warnings.warn("Weight dropout not implemented yet. Ignoring them.", RuntimeWarning)
        return constants

    def get_config(self):
        # This function is called to get the model configuration while serializing it
        # Essentially has all the arguments in the __init__ method as a dict.
        config = {'stack_limit': self.stack_limit,
                  'buffer_ops_limit': self.buffer_ops_limit,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation':self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizers[0].get_config() if self.W_regularizers else None,
                  'U_regularizer': self.U_regularizers[0].get_config() if self.U_regularizers else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizers[0].get_config() if self.b_regularizers else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_V': self.dropout_V}
        base_config = super(TreeCompositionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BOWEncoder(Layer):
    '''
    Bag of Words Encoder takes a matrix of shape (num_words, word_dim) and returns a vector of size (word_dim),
    which is an average of the (unmasked) rows in the input matrix. This could have been done using a Lambda
    layer, except that Lambda layer does not support masking (as of Keras 1.0.7).
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]

        # For consistency of handling sentence encoders, we will often get passed this parameter.
        # We don't use it, but Layer will complain if it's there, so we get rid of it here.
        kwargs.pop('output_dim', None)
        super(BOWEncoder, self).__init__(**kwargs)

    @overrides
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])  # removing second dimension

    @overrides
    def call(self, x, mask=None):
        # pylint: disable=redefined-variable-type
        if mask is None:
            return K.mean(x, axis=1)
        else:
            # Compute weights such that masked elements have zero weights and the remaining
            # weight is ditributed equally among the unmasked elements.
            # Mask (samples, num_words) has 0s for masked elements and 1s everywhere else.
            # Mask is of type int8. While theano would automatically make weighted_mask below
            # of type float32 even if mask remains int8, tensorflow would complain. Let's cast it
            # explicitly to remain compatible with tf.
            float_mask = K.cast(mask, 'float32')
            # Expanding dims of the denominator to make it the same shape as the numerator.
            weighted_mask = float_mask / K.expand_dims(K.sum(float_mask, axis=1))  # (samples, num_words)
            weighted_mask = K.expand_dims(weighted_mask)  # (samples, num_words, 1)
            return K.sum(x * weighted_mask, axis=1)  # (samples, word_dim)

    @overrides
    def compute_mask(self, input, input_mask=None):  # pylint: disable=redefined-builtin
        # We need to override this method because Layer passes the input mask unchanged since this layer
        # supports masking. We don't want that. After the input is averaged, we can stop propagating
        # the mask.
        return None




class PositionalEncoder(Layer):
    '''
    A Positional Encoder takes a matrix of shape (num_words, word_dim) and returns a vector of size (word_dim),
    which implements the following linear combination of the rows:

     representation = sum_(j=1)^(n) { l_j * w_j }

     where w_j is the j-th word representation in the sentence and l_j is a vector defined as follows:

     l_kj =  (1 - j)/m  -  (k/d)((1-2j)/m)

     where:
      - j is the word sentence index
      - m is the sentence length
      - k is the vector index(ie the k-th element of a vector)
      - d is the dimension of the embedding.

    This could have been done using a Lambda
    layer, except that Lambda layer does not support masking (as of Keras 1.0.7).
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        super(PositionalEncoder, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])  # removing second dimension

    def call(self, x, mask=None):
        # pylint: disable=redefined-variable-type
        def my_keras_cumsum(x, axis=0):
            """
            Keras doesn't have a cumsum operation yet, but it seems to be nearly there - see this PR:
             https://github.com/fchollet/keras/pull/3791.
             """
            if K.backend() == "tensorflow":
                return tf.cumsum(x, axis=axis)
            else:
                return T.cumsum(x, axis=axis)

        # This section implements the positional encoder on all the vectors at once.
        # The general idea is to use ones matrices in the shape of x to create indexes per word.

        if mask is None:
            ones_like_x = K.ones_like(x)
        else:
            float_mask = K.cast(mask, 'float32')
            ones_like_x = K.ones_like(x) * K.expand_dims(float_mask, 2)

        # This is an odd way to get the number of words(ie the first dimension of x).
        # However, if the input is masked, using the dimension directly does not
        # equate to the correct number of words. We fix this by adding up a relevant
        # row of ones which has been masked if required.
        masked_m = K.expand_dims(K.sum(ones_like_x, 1), 1)

        one_over_m = ones_like_x / masked_m
        j_index = my_keras_cumsum(ones_like_x, 1)
        d_over_D = my_keras_cumsum(ones_like_x, 2) * 1.0/K.cast(K.shape(x)[2], 'float32')
        one_minus_j = ones_like_x - j_index
        one_minus_two_j = ones_like_x - 2 * j_index

        l_weighting_vectors = (one_minus_j * one_over_m) - \
                              (d_over_D * (one_minus_two_j * one_over_m))

        return l_weighting_vectors * x

    def compute_mask(self, input, input_mask=None):  # pylint: disable=redefined-builtin
        # We need to override this method because Layer passes the input mask unchanged since this layer
        # supports masking. We don't want that. After the input is merged we can stop propagating
        # the mask.
        return None


class CNNEncoder(Layer):
    '''
    CNNEncoder is a combination of multiple convolution layers and max pooling layers. This is
    defined as a single layer to be consistent with the other encoders in terms of input and output
    specifications.  The input to this "layer" is of shape (batch_size, num_words, embedding_size)
    and the output is of size (batch_size, output_dim).

    The CNN has one convolution layer per each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    depends on the ngram size: input_length - ngram_size + 1. The corresponding maxpooling layer
    aggregates all these outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is len(ngram_filter_sizes) * num_filters.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.
    '''
    def __init__(self, weights=None, **kwargs):
        self.supports_masking = True

        # This is the output dim for each convolutional layer, which is the same as the number of
        # "filters" learned by that layer.
        self.num_filters = kwargs.pop('num_filters')

        # This specifies both the number of convolutional layers we will create and their sizes.
        # Must be a List[int].  The default of (2, 3, 4, 5) will have four convolutional layers,
        # corresponding to encoding ngrams of size 2 to 5 with some number of filters.
        ngram_filter_sizes = kwargs.pop('ngram_filter_sizes', (2, 3, 4, 5))
        self.ngram_filter_sizes = ngram_filter_sizes

        self.output_dim = kwargs.pop('output_dim')

        conv_layer_activation = kwargs.pop('conv_layer_activation', 'relu')
        self.conv_layer_activation = conv_layer_activation

        self.W_regularizer = kwargs.pop("W_regularizer", None)  # pylint: disable=invalid-name
        self.b_regularizer = kwargs.pop("b_regularizer", None)

        # These are member variables that will be defined during self.build().
        self.convolution_layers = None
        self.max_pooling_layers = None
        self.projection_layer = None

        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        super(CNNEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        input_length = input_shape[1]  # number of words
        input_dim = input_shape[-1]
        # We define convolution, maxpooling and dense layers first.
        self.convolution_layers = [Convolution1D(nb_filter=self.num_filters,
                                                 filter_length=ngram_size,
                                                 activation=self.conv_layer_activation,
                                                 W_regularizer=self.W_regularizer,
                                                 b_regularizer=self.b_regularizer)
                                   for ngram_size in self.ngram_filter_sizes]
        self.max_pooling_layers = [MaxPooling1D(pool_length=input_length - ngram_size + 1)
                                   for ngram_size in self.ngram_filter_sizes]
        self.projection_layer = Dense(input_dim)
        # Building all layers because these sub-layers are not explitly part of the computatonal graph.
        for convolution_layer, max_pooling_layer in zip(self.convolution_layers, self.max_pooling_layers):
            convolution_layer.build(input_shape)
            max_pooling_layer.build(convolution_layer.get_output_shape_for(input_shape))
        maxpool_output_dim = self.num_filters * len(self.ngram_filter_sizes)
        projection_input_shape = (input_shape[0], maxpool_output_dim)
        self.projection_layer.build(projection_input_shape)
        # Defining the weights of this "layer" as the set of weights from all convolution
        # and maxpooling layers.
        self.trainable_weights = []
        for layer in self.convolution_layers + self.max_pooling_layers + [self.projection_layer]:
            self.trainable_weights.extend(layer.trainable_weights)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        # Each convolution layer returns output of size (samples, pool_length, num_filters),
        #       where pool_length = num_words - ngram_size + 1
        # Each maxpooling layer returns output of size (samples, 1, num_filters).
        # We need to flatten to remove the second dimension of length 1 from the maxpooled output.
        filter_outputs = [K.batch_flatten(max_pooling_layer.call(convolution_layer.call(x, mask)))
                          for max_pooling_layer, convolution_layer in zip(self.max_pooling_layers,
                                                                          self.convolution_layers)]
        maxpool_output = merge(filter_outputs, mode='concat') if len(filter_outputs) > 1 else filter_outputs[0]
        return self.projection_layer.call(maxpool_output)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def compute_mask(self, input, input_mask=None):  # pylint: disable=redefined-builtin
        # By default Keras propagates the mask from a layer that supports masking. We don't need it
        # anymore. So eliminating it from the flow.
        return None

    def get_config(self):
        config = {"num_filters": self.num_filters,
                  "ngram_filter_sizes": self.ngram_filter_sizes,
                  "conv_layer_activation": self.conv_layer_activation,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None
                 }
        base_config = super(CNNEncoder, self).get_config()
        config.update(base_config)
        return config


def set_regularization_params(encoder_type: str, params: Dict[str, Any]):
    """
    This method takes regularization parameters that are specified in `params` and converts them
    into Keras regularization objects, modifying `params` to contain the correct keys for the given
    encoder_type.

    Currently, we only allow specifying a consistent regularization across all the weights of a
    layer.
    """
    l1_regularization = params.pop("l1_regularization", None)
    l2_regularization = params.pop("l2_regularization", None)
    regularizer = lambda: l1l2(l1=l1_regularization, l2=l2_regularization)
    if encoder_type == 'cnn':
        params["W_regularizer"] = regularizer()
        params["b_regularizer"] = regularizer()
    elif encoder_type == 'lstm':
        params["W_regularizer"] = regularizer()
        params["U_regularizer"] = regularizer()
        params["b_regularizer"] = regularizer()
    elif encoder_type == 'tree_lstm':
        params["W_regularizer"] = regularizer()
        params["U_regularizer"] = regularizer()
        params["V_regularizer"] = regularizer()
        params["b_regularizer"] = regularizer()
    return params


# The first item added here will be used as the default in some cases.
encoders = OrderedDict()  # pylint:  disable=invalid-name
encoders["bow"] = BOWEncoder
encoders["lstm"] = LSTM
encoders["tree_lstm"] = TreeCompositionLSTM
encoders["cnn"] = CNNEncoder
encoders["positional"] = PositionalEncoder
