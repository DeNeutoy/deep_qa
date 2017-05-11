# pylint: disable=no-self-use,invalid-name


import tensorflow
import keras.backend as K
from deep_qa.common.params import Params, pop_choice
from deep_qa.models.text_classification import ClassificationModel
from ..common.test_case import DeepQaTestCase


class TestMultiGpu(DeepQaTestCase):
    # pylint: disable=protected-access

    def setUp(self):
        super(TestMultiGpu, self).setUp()
        self.write_true_false_model_files()
        self.args = Params({
                    'embedding_dim': {'words': 4, 'characters': 2},
                    'batch_size': 8,
                    'num_gpus': 2,
                    'save_models': True,
                    'show_summary_with_masking_info': True,
            })

    def test_model_can_train(self):

        model = self.get_model(ClassificationModel, self.args)
        model.train()

    def test_multi_gpu_shares_variables(self):

        multi_gpu_model = self.get_model(ClassificationModel, self.args)
        self.args['num_gpus'] = 1
        single_gpu_model = self.get_model(ClassificationModel, self.args)

        multi_gpu_model.train()
        multi_gpu_variables = [x.name for x in multi_gpu_model.model.trainable_weights]

        K.clear_session()
        single_gpu_model.train()
        single_gpu_variables = [x.name for x in single_gpu_model.model.trainable_weights]

        print("Single GPU:", single_gpu_variables)
        print("Multi GPU:", multi_gpu_variables)
        assert single_gpu_variables == multi_gpu_variables







