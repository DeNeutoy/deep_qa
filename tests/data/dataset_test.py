# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy

from deep_qa.data.fields import TextField, LabelField
from deep_qa.data.instance import Instance
from deep_qa.data.dataset import Dataset
from deep_qa.data.vocabulary import Vocabulary
from deep_qa.data.token_indexers import token_indexers
from deep_qa.common.checks import ConfigurationError
from deep_qa.testing.test_case import DeepQaTestCase


class TestDataset(DeepQaTestCase):

    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("this", "words")
        self.vocab.add_token_to_namespace("is", "words")
        self.vocab.add_token_to_namespace("a", "words")
        self.vocab.add_token_to_namespace("sentence", "words")
        self.vocab.add_token_to_namespace(".", "words")
        super(TestDataset, self).setUp()

    def test_instances_must_have_homogeneous_fields(self):
        instance1 = Instance({"tag": (LabelField(1))})
        instance2 = Instance({"words": TextField(["hello"], [])})
        with pytest.raises(ConfigurationError):
            _ = Dataset([instance1, instance2])

    def test_padding_lengths_uses_max_instance_lengths(self):

        dataset = self.get_dataset()
        dataset.index_instances(self.vocab)

        padding_lengths = dataset.get_padding_lengths()
        assert padding_lengths == {"text1": {"num_tokens": 5}, "text2": {"num_tokens": 6}}

    def test_as_arrays(self):
        dataset = self.get_dataset()
        dataset.index_instances(self.vocab)
        padding_lengths = dataset.get_padding_lengths()
        arrays = dataset.as_arrays(padding_lengths)

        text1 = arrays["text1"][0]
        text2 = arrays["text2"][0]
        numpy.testing.assert_array_almost_equal(text1, numpy.array([[2, 3, 4, 5, 6],
                                                                     [1, 3, 4, 5, 6]]))

        numpy.testing.assert_array_almost_equal(text2, numpy.array([[2, 3, 4, 1, 5, 6],
                                                                    [2, 3, 1, 0, 0, 0]]))

    def get_dataset(self):
        field1 = TextField(["this", "is", "a", "sentence", "."], [token_indexers["single id"]("words")])
        field2 = TextField(["this", "is", "a", "different", "sentence", "."], [token_indexers["single id"]("words")])
        field3 = TextField(["here", "is", "a", "sentence", "."], [token_indexers["single id"]("words")])
        field4 = TextField(["this", "is", "short"], [token_indexers["single id"]("words")])
        instances = [Instance({"text1": field1, "text2": field2}),
                     Instance({"text1": field3, "text2": field4})]

        return Dataset(instances)