# pylint: disable=no-self-use,invalid-name
from deep_qa.data.dataset_readers import SnliReader
from deep_qa.testing.test_case import DeepQaTestCase


class TestSnliDataset(DeepQaTestCase):

    def setUp(self):
        super(TestSnliDataset, self).setUp()
        self.write_original_snli_data()

    def test_read_from_file(self):

        reader = SnliReader(self.TRAIN_FILE)
        dataset = reader.read()

        instance1 = {"premise": ["a", "person", "on", "a", "horse",
                                 "jumps", "over", "a", "broken", "down", "airplane", "."],
                     "hypothesis": ["a", "person", "is", "training",
                                    "his", "horse", "for", "a", "competition", "."],
                     "label": "neutral"}

        instance2 = {"premise": ["a", "person", "on", "a", "horse",
                                 "jumps", "over", "a", "broken", "down", "airplane", "."],
                     "hypothesis": ["a", "person", "is", "at", "a", "diner",
                                    ",", "ordering", "an", "omelette", "."],
                     "label": "contradiction"}
        instance3 = {"premise": ["a", "person", "on", "a", "horse",
                                 "jumps", "over", "a", "broken", "down", "airplane", "."],
                     "hypothesis": ["a", "person", "is", "outdoors", ",", "on", "a", "horse", "."],
                     "label": "entailment"}

        assert len(dataset.instances) == 3
        fields = dataset.instances[0].fields()
        assert fields["premise"].tokens() == instance1["premise"]
        assert fields["hypothesis"].tokens() == instance1["hypothesis"]
        assert fields["label"].label() == instance1["label"]
        fields = dataset.instances[1].fields()
        assert fields["premise"].tokens() == instance2["premise"]
        assert fields["hypothesis"].tokens() == instance2["hypothesis"]
        assert fields["label"].label() == instance2["label"]
        fields = dataset.instances[2].fields()
        assert fields["premise"].tokens() == instance3["premise"]
        assert fields["hypothesis"].tokens() == instance3["hypothesis"]
        assert fields["label"].label() == instance3["label"]
