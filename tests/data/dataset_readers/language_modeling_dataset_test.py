# pylint: disable=no-self-use,invalid-name
from deep_qa.data.dataset_readers import LanguageModelingReader
from deep_qa.testing.test_case import DeepQaTestCase


class TestLanguageModellingDatasetReader(DeepQaTestCase):

    def setUp(self):
        super(TestLanguageModellingDatasetReader, self).setUp()
        self.write_sentence_data()

    def test_read_from_file(self):
        reader = LanguageModelingReader(self.TRAIN_FILE,
                                        tokens_per_instance=4)

        dataset = reader.read()
        instances = dataset.instances
        assert instances[0].fields()["input_tokens"].tokens() == ["<S>", "this", "is", "a", "sentence"]
        assert instances[1].fields()["input_tokens"].tokens() == ["<S>", "for", "language", "modelling", "."]
        assert instances[2].fields()["input_tokens"].tokens() == ["<S>", "here", "'s", "another", "one"]
