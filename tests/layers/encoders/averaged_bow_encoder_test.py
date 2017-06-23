# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input, Embedding
from keras.models import Model

from deep_qa.layers.wrappers.output_mask import OutputMask
from deep_qa.layers.encoders.AveragedBOWEncoder import AveragedBOWEncoder
from tests.common.test_case import DeepQaTestCase


class TestAveragedBOWEncoder(DeepQaTestCase):

    def test_on_unmasked_input(self):
        dimension_to_average = 2
        num_dimensions = 3
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding does not mask zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)
        encoder = AveragedBOWEncoder(dimension_to_average, num_dimensions)
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        expected_output = numpy.mean(embedding_weights[test_input], axis=dimension_to_average)
        actual_output = model.predict(test_input)
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_input(self):
        # Average over a dimension in which some elements are masked, and
        # check that they are masked correctly in the average.
        dimension_to_average = 1
        num_dimensions = 3
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder(dimension_to_average, num_dimensions)
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        encoder_mask = OutputMask()(encoded_input)
        model = Model(inputs=input_layer, outputs=[encoded_input, encoder_mask])
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.

        # Don't take the first element because it should be masked.
        expected_output = numpy.mean(embedding_weights[test_input[:, 1:]], axis=dimension_to_average)
        actual_output, actual_mask = model.predict(test_input)
        # Mask should now

        numpy.testing.assert_array_equal(actual_mask, numpy.array([True]))
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_mask_is_propagated_if_required(self):
        # Here we test averaging over a dimension which is not masked, but in which the
        # output still requires a mask.
        dimension_to_average = 2
        num_dimensions = 3
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder(dimension_to_average, num_dimensions)
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        encoder_mask = OutputMask()(encoded_input)
        model = Model(inputs=input_layer, outputs=[encoded_input, encoder_mask])
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.

        # Here, the dimension we are reducing is the embedding dimension. In this case,
        # the actual value of the returned output should be equal to averaging without masking,
        # (as there is nothing to mask in a dimension not covered by the mask) but the mask should
        # be propagated through the layer, still masking the correct index.
        expected_output = numpy.mean(embedding_weights[test_input], axis=dimension_to_average)
        actual_output, actual_mask = model.predict(test_input)
        # First index should still be masked.
        numpy.testing.assert_array_equal(actual_mask, numpy.array([[False, True, True, True, True]]))
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_all_zeros(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        encoder = AveragedBOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 0, 0, 0, 0]], dtype='int32')
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.zeros((1, embedding_dim))
        actual_output = model.predict(test_input)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)
