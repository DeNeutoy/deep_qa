from typing import List

from keras import backend as K
from overrides import overrides

from ...common.checks import ConfigurationError
from .similarity_function import SimilarityFunction


class DotProduct(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors.  It has
    no parameters.
    """
    def __init__(self, **kwargs):
        super(DotProduct, self).__init__(**kwargs)

    @overrides
    def initialize_weights(self, tensor_1_dim: int, tensor_2_dim: int) -> List['K.variable']:
        if tensor_1_dim != tensor_2_dim:
            raise ConfigurationError("Tensor dims must match for dot product similarity, but "
                                     "were {} and {}".format(tensor_1_dim, tensor_2_dim))
        return []

    @overrides
    def compute_similarity(self, tensor_1, tensor_2):
        return K.sum(tensor_1 * tensor_2, axis=-1)
