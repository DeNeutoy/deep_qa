# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest

from deep_qa.data.vocabulary import Vocabulary
from deep_qa.data.fields import TextField, TagField
from deep_qa.data.token_indexers import token_indexers
from deep_qa.common.checks import ConfigurationError
from deep_qa.testing.test_case import DeepQaTestCase


class TestTextField(DeepQaTestCase):

    def test_field_counts_vocab_items_correctly(self):
        field = TextField(["This", "is", "a", "text", "field", "."],
                          token_indexers=[token_indexers["single id"]("words")])
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

    def test_tag_length_mismatch_raises(self):

        with pytest.raises(ConfigurationError):

            text = TextField(["here", "are", "some", "words", "."], [])
            wrong_tags = ["B", "O", "O"]
            _ = TagField(wrong_tags, text)


    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        sentence_index = vocab.add_token_to_namespace("sentence", namespace='words')
        capital_a_index = vocab.add_token_to_namespace("A", namespace='words')
        capital_a_char_index = vocab.add_token_to_namespace("A", namespace='characters')
        s_index = vocab.add_token_to_namespace("s", namespace='characters')
        e_index = vocab.add_token_to_namespace("e", namespace='characters')
        n_index = vocab.add_token_to_namespace("n", namespace='characters')
        t_index = vocab.add_token_to_namespace("t", namespace='characters')
        c_index = vocab.add_token_to_namespace("c", namespace='characters')

        field = TextField(["A", "sentence"], [token_indexers["single id"](token_namespace="words")])
        field.index(vocab)
        assert field._indexed_tokens == [[capital_a_index, sentence_index]]

        field1 = TextField(["A", "sentence"], [token_indexers["characters"](character_namespace="characters")])
        field1.index(vocab)
        assert field1._indexed_tokens == [[[capital_a_char_index], [s_index, e_index, n_index, t_index,
                                           e_index, n_index, c_index, e_index]]]
        field2 = TextField(["A", "sentence"],
                           token_indexers=[token_indexers["single id"](token_namespace="words"),
                                           token_indexers["characters"](character_namespace="characters")])
        field2.index(vocab)
        assert field2._indexed_tokens == [[capital_a_index, sentence_index],
                                          [[capital_a_char_index],
                                           [s_index, e_index, n_index, t_index,
                                            e_index, n_index, c_index, e_index]]
                                          ]