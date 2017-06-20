# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data.vocabulary import Vocabulary
from deep_qa.data.fields import LabelField
from deep_qa.testing.test_case import DeepQaTestCase


class TestLabelField(DeepQaTestCase):

    def test_pad_returns_one_hot_array(self):
        label = LabelField(5, num_labels=10)
        array = label.pad(label.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(array[0], numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

    def test_label_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("entailment", namespace="*labels")
        vocab.add_token_to_namespace("contradiction", namespace="*labels")
        vocab.add_token_to_namespace("neutral", namespace="*labels")

        label = LabelField("entailment")
        label.index(vocab)
        array = label.pad(label.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(array[0], numpy.array([1, 0, 0]))
