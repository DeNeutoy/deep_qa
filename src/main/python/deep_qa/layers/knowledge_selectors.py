'''
Knowledge selectors take an encoded sentence (or logical form) representation and encoded
representations of background facts related to the sentence, and compute an attention over the
background representations. By default, the attention is soft (the attention values are in the
range (0, 1)). But we can optionally pass 'hard_selection=True' to the constructor, to make it
hard (values will be all 0, except one).
'''

from collections import OrderedDict
from overrides import overrides

from keras.engine import InputSpec
from keras import backend as K
from keras import activations, initializations
from keras.layers import Layer
from keras.layers.recurrent import GRU

def tile_sentence_encoding(sentence_encoding, knowledge_encoding):
    # Tensorflow can't use unknown sizes at runtime, so we have to make use of the broadcasting
    # ability of TF and Theano instead to create the tiled sentence encoding.

    # Shape: (knowledge_length, num_samples, encoding_dim)
    k_ones = K.permute_dimensions(K.ones_like(knowledge_encoding), [1, 0, 2])
    # Now we have a (knowledge_length, num_samples, encoding_dim)*(num_samples, encoding_dim)
    # elementwise multiplication which is broadcast. We then reshape back.
    tiled_sentence_encoding = K.permute_dimensions(k_ones * sentence_encoding, [1, 0, 2])
    return tiled_sentence_encoding


def hardmax(unnormalized_attention, knowledge_length):
    # (num_samples, knowledge_length)
    tiled_max_values = K.tile(K.expand_dims(K.max(unnormalized_attention, axis=1)), (1, knowledge_length))
    # We now have a matrix where every column in each row has the max knowledge score value from
    # the corresponding row in the unnormalized attention matrix.  Next, we will compare that
    # all-max matrix with the original input, resulting in ones where the column equals max and
    # zero everywhere else.
    # Shape: (num_samples, knowledge_length)
    bool_max_attention = K.equal(unnormalized_attention, tiled_max_values)
    # Needs to be cast to be compatible with TensorFlow
    max_attention = K.cast(bool_max_attention, 'float32')
    return max_attention


class DotProductKnowledgeSelector(Layer):
    """
    Input Shape: num_samples, (knowledge_length + 1), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the sentence, i[:, 1:, :]
    are the encodings of the background facts.  There is no need to specify knowledge length here.

    Attend to facts conditioned on the input sentence, just using a dot product between the input
    vector and the background vectors (i.e., there are no parameters here).  This layer is a
    reimplementation of the memory layer in "End-to-End Memory Networks", Sukhbaatar et al. 2015.
    """
    def __init__(self, hard_selection=False, **kwargs):
        self.input_spec = [InputSpec(ndim=3)]
        self.hard_selection = hard_selection
        super(DotProductKnowledgeSelector, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # Assumption: The first row in each slice corresponds to the encoding of the input and the
        # remaining rows to those of the background knowledge.

        sentence_encoding = x[:, 0, :]  # (num_samples, input_dim)
        knowledge_encoding = x[:, 1:, :]  # (num_samples, knowledge_length, input_dim)

        # We want to take a dot product of the knowledge matrix and the sentence vector from each
        # sample. Instead of looping over all samples (inefficient), let's tile the sentence
        # encoding to make it the same size as knowledge encoding, take an element wise product and
        # sum over the last dimension (dim = 2).

        # (num_samples, knowledge_length, input_dim)
        tiled_sentence_encoding = tile_sentence_encoding(sentence_encoding, knowledge_encoding)

        # (num_samples, knowledge_length)
        unnormalized_attention = K.sum(knowledge_encoding * tiled_sentence_encoding, axis=2)
        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = K.softmax(unnormalized_attention)
        return knowledge_attention

    def get_output_shape_for(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 1)  # (num_samples, knowledge_length)


class ParameterizedKnowledgeSelector(Layer):
    """
    Here we are reimplementing the attention part of the memory layer described in
    "Teaching Machines to Read and Comprehend", Hermann et al., 2015.

    Input Shape: num_samples, (knowledge_length + 1), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the sentence, i[:, 1:, :]
    are the encodings of the background facts.  There is no need to specify knowledge length here.

    This layer concatenates the input with each background sentence, passes them through a
    non-linearity, then does a softmax to get attention weights.

    Equations:
    Inputs: u is the sentence encoding, z_t are the background sentence encodings
    Weights: W_1 (called self.dense_weights), v (called self.dot_bias)
    Output: a_t

    m_t = tanh(W_1 * concat(z_t, u))
    q_t = dot(v, m_t)
    a_t = softmax(q_t)
    """

    def __init__(self,
                 activation='tanh',
                 initialization='glorot_uniform',
                 hard_selection=False,
                 weights=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.init = initializations.get(initialization)
        self.hard_selection = hard_selection
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.dense_weights = None
        self.dot_bias = None
        super(ParameterizedKnowledgeSelector, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.dense_weights = self.init((input_dim * 2, input_dim), name='{}_dense'.format(self.name))
        self.dot_bias = self.init((input_dim, 1), name='{}_dot_bias'.format(self.name))
        self.trainable_weights = [self.dense_weights, self.dot_bias]

        # Now that trainable_weights is complete, we set weights if needed.
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        '''
        Equations repeated from above:
        Inputs: u is the sentence encoding, z_t are the background sentence encodings
        Weights: W_1 (called self.dense_weights), v (called self.dot_bias)
        Output: a_t

        (1) zu_t = concat(z_t, u)
        (2) m_t = tanh(dot(W_1, zu_t))
        (3) q_t = dot(v, m_t)
        (4) a_t = softmax(q_t)

        Here we actually implement the logic of these equations.  We label each step with its
        number and the variable above that it's computing.  The implementation looks more complex
        than these equations because we have to unpack the input, then use tiling instead of loops
        to make this more efficient.
        '''
        # Remember that the first row in each slice corresponds to the encoding of the input and
        # the remaining rows to those of the background knowledge.
        sentence_encoding = x[:, 0, :]  # (num_samples, input_dim)
        knowledge_encoding = x[:, 1:, :]  # (num_samples, knowledge_length, input_dim)

        # We're going to have to do several operations on the input sentence for each background
        # sentence.  Instead of looping over the background sentences, which is inefficient, we'll
        # tile the sentence encoding and use it in what follows.

        # (num_samples, knowledge_length, input_dim)
        tiled_sentence_encoding = tile_sentence_encoding(sentence_encoding, knowledge_encoding)

        # (1: zu_t) Result of this is (num_samples, knowledge_length, input_dim * 2)
        concatenated_encodings = K.concatenate([knowledge_encoding, tiled_sentence_encoding])

        # (2: m_t) Result of this is (num_samples, knowledge_length, input_dim)
        concatenated_activation = self.activation(K.dot(concatenated_encodings, self.dense_weights))

        # (3: q_t) Result of this is (num_samples, knowledge_length).  We need to remove a dimension
        # after the dot product with K.squeeze, otherwise this would be (num_samples,
        # knowledge_length, 1), which is not a valid input to K.softmax.
        unnormalized_attention = K.squeeze(K.dot(concatenated_activation, self.dot_bias), axis=2)

        # (4: a_t) Result is (num_samples, knowledge_length)
        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = K.softmax(unnormalized_attention)
        return knowledge_attention

    def get_output_shape_for(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 1)  # (num_samples, knowledge_length)


class AttentionBasedGRUKnowledgeSelector(Layer):
    '''
    Input Shape: num_samples, (knowledge_length + 1), input_dim.

    This Knowledge Selector runs an Attentive GRU over the knowledge

    '''
    def __init__(self, encoding_dim, name="sum_memory_updater", **kwargs):
        super(AttentionBasedGRUKnowledgeSelector, self).__init__(name=name, **kwargs)
        self.encoding_dim = encoding_dim
        self.mode = 'sum'
        self.attentive_GRU =

    def call(self, x, mask=None):
        memory_vector = x[:, :self.encoding_dim]
        aggregated_knowledge_vector = x[:, self.encoding_dim:]
        return memory_vector + aggregated_knowledge_vector

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], int(input_shape[1] / 2))

    def get_config(self):
        base_config = super(AttentionBasedGRUKnowledgeSelector, self).get_config()
        config = {'encoding_dim': self.encoding_dim}
        config.update(base_config)
        return config


class AttentiveGRU(GRU):

    def __init__(self, attention_output, encoding_dim, name="attentive_gru", **kwargs):
        super(AttentiveGRU, self).__init__(encoding_dim, name=name, **kwargs)
        self.attention_output = attention_output

    @overides
    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))

        # TODO Check: Is this right in Keras???
        h = z * hh + (1 - z) * h_tm1
        return h, [h]






# The first item added here will be used as the default in some cases.
selectors = OrderedDict()  # pylint: disable=invalid-name
selectors['parameterized'] = ParameterizedKnowledgeSelector
selectors['dot_product'] = DotProductKnowledgeSelector
