# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from deep_qa.data.token_indexers import token_indexers
from deep_qa.data.tokenizers.character_tokenizer import CharacterTokenizer
from deep_qa.data.vocabulary import Vocabulary
from deep_qa.testing.test_case import DeepQaTestCase


class CharacterTokenIndexerTest(DeepQaTestCase):

    def test_count_vocab_items_respects_casing(self):
        indexer = token_indexers["characters"]("characters")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items("Hello", counter)
        indexer.count_vocab_items("hello", counter)
        assert counter["characters"] == {"h": 1, "H": 1, "e": 2, "l": 4, "o": 2}

        indexer = token_indexers["characters"]("characters",
                                               CharacterTokenizer(lowercase_characters=True))
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items("Hello", counter)
        indexer.count_vocab_items("hello", counter)
        assert counter["characters"] == {"h": 2, "e": 2, "l": 4, "o": 2}

    def test_pad_token_sequence(self):
        indexer = token_indexers["characters"]("characters")
        padded_tokens = indexer.pad_token_sequence([[1, 2, 3, 4, 5], [1, 2, 3], [1]],
                                                   desired_num_tokens=4,
                                                   padding_lengths={"num_token_characters": 10})
        assert padded_tokens == [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                 [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_token_to_indices_produces_correct_characters(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("A", namespace='characters')
        vocab.add_token_to_namespace("s", namespace='characters')
        vocab.add_token_to_namespace("e", namespace='characters')
        vocab.add_token_to_namespace("n", namespace='characters')
        vocab.add_token_to_namespace("t", namespace='characters')
        vocab.add_token_to_namespace("c", namespace='characters')

        indexer = token_indexers["characters"]("characters")
        indices = indexer.token_to_indices("sentential", vocab)
        assert indices == [3, 4, 5, 6, 4, 5, 6, 1, 1, 1]
