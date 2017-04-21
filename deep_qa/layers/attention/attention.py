from copy import deepcopy
from typing import Dict

from keras import backend as K
from overrides import overrides

from ...common.params import pop_choice_with_default
from ..masked_layer import MaskedLayer
from ...tensors.masked_operations import masked_softmax
from ...tensors.similarity_functions import similarity_functions


class Attention(MaskedLayer):
    """
    This Layer takes two inputs: a vector and a matrix.  We compute the
    similarity between the vector and each row in the matrix, and then perform
    a softmax over rows using those computed similarities.  We handle masking
    properly for masked rows in the matrix, though we ignore any masking on
    the vector.

    By default similarity is computed with a dot product, but you can
    alternatively use a parameterized similarity function if you wish.

    Inputs:

    - vector: shape ``(batch_size, embedding_dim)``, mask is ignored if provided
    - matrix: shape ``(batch_size, num_rows, embedding_dim)``, with mask ``(batch_size, num_rows)``

    Output:

    - attention: shape ``(batch_size, num_rows)``, no mask (masked input rows have value 0 in the
      output)

    Parameters
    ----------
    similarity_function_params: Dict, optional (default={})
        These parameters get passed to a similarity function (see
        :mod:`deep_qa.tensors.similarity_functions` for more info on what's acceptable).  The
        default similarity function with no parameters is a simple dot product.
    """
    # pylint: disable=dangerous-default-value
    def __init__(self, similarity_function: Dict={}, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.similarity_function_params = deepcopy(similarity_function)
        sim_function_choice = pop_choice_with_default(similarity_function,
                                                      'type', list(similarity_functions.keys()))
        # TODO(mark): The default value here is mutable so this will introduce naming bugs
        # if it is called more than once with default parameters.
        similarity_function['name'] = self.name + '_similarity_function'
        self.similarity_function = similarity_functions[sim_function_choice](**similarity_function)

    @overrides
    def build(self, input_shape):
        tensor_1_dim = input_shape[0][-1]
        tensor_2_dim = input_shape[1][-1]
        self.trainable_weights = self.similarity_function.initialize_weights(tensor_1_dim, tensor_2_dim)
        super(Attention, self).build(input_shape)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    @overrides
    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1])

    @overrides
    def call(self, inputs, mask=None):
        vector, matrix = inputs
        if mask is None:
            matrix_mask = None
        else:
            matrix_mask = mask[1]
        num_rows = K.int_shape(matrix)[1]
        tiled_vector = K.repeat_elements(K.expand_dims(vector, axis=1), num_rows, axis=1)
        similarities = self.similarity_function.compute_similarity(tiled_vector, matrix)
        return masked_softmax(similarities, matrix_mask)

    @overrides
    def get_config(self):
        base_config = super(Attention, self).get_config()
        config = {'similarity_function': self.similarity_function_params}
        config.update(base_config)
        return config
