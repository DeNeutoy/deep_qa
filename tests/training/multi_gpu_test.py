# pylint: disable=no-self-use,invalid-name


from copy import deepcopy
import keras.backend as K
import tensorflow
from deep_qa.training.multi_gpu import pin_variable_device_scope
from deep_qa.common.params import Params
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

    def test_model_can_train_and_load(self):
        self.ensure_model_trains_and_loads(ClassificationModel, self.args)

    def test_pinned_scope_correctly_allocates_ops(self):

        scope_function = pin_variable_device_scope(device="/gpu:0", parameter_device="/cpu:0")

        # Should have a cpu scope.
        variable = tensorflow.Variable([])
        # Should have a gpu scope.
        add_op = tensorflow.add(variable, 1.0)

        assert scope_function(variable.op) == "/cpu:0"
        assert scope_function(add_op.op) == "/gpu:0"

    def test_variables_live_on_cpu(self):

        model = self.get_model(ClassificationModel, self.args)
        model.train()

        print(model.model.layers)
        trainable_variables = model.model.trainable_weights

        for variable in trainable_variables:
            assert variable.device == "/cpu:0" or variable.device == ""

    def test_multi_gpu_shares_variables(self):

        multi_gpu_model = self.get_model(ClassificationModel, self.args)

        single_gpu_args = deepcopy(self.args)
        single_gpu_args["num_gpus"] = 1
        single_gpu_model = self.get_model(ClassificationModel, self.args)

        multi_gpu_model.train()
        multi_gpu_variables = [x.name for x in multi_gpu_model.model.trainable_weights]

        K.clear_session()
        single_gpu_model.train()
        single_gpu_variables = [x.name for x in single_gpu_model.model.trainable_weights]

        print("Single GPU:", single_gpu_variables)
        print("Multi GPU:", multi_gpu_variables)
        assert single_gpu_variables == multi_gpu_variables







