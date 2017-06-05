# pylint: disable=no-self-use,invalid-name
from copy import deepcopy

import keras.backend as K
import tensorflow
import numpy

from deep_qa.training.train_utils import pin_variable_device_scope
from deep_qa.training.train_utils import _get_dense_gradient_average, _get_sparse_gradient_average
from deep_qa.common.params import Params
from deep_qa.models.text_classification import ClassificationModel
from ..common.test_case import DeepQaTestCase


class TestMultiGpu(DeepQaTestCase):

    def setUp(self):
        super(TestMultiGpu, self).setUp()
        self.write_true_false_model_files()
        self.args = Params({
                'num_gpus': 2,
        })

    def test_model_can_train_and_load(self):
        self.ensure_model_trains_and_loads(ClassificationModel, self.args)

    def test_pinned_scope_correctly_allocates_ops(self):
        scope_function = pin_variable_device_scope(device="/gpu:0", variable_device="/cpu:0")

        # Should have a cpu scope.
        variable = tensorflow.Variable([])
        # Should have a gpu scope.
        add_op = tensorflow.add(variable, 1.0)

        assert scope_function(variable.op) == "/cpu:0"
        assert scope_function(add_op.op) == "/gpu:0"  # pylint: disable=no-member

    def test_variables_live_on_cpu(self):
        model = self.get_model(ClassificationModel, self.args)
        model.train()

        trainable_variables = model.model.trainable_weights
        for variable in trainable_variables:
            # This is an odd quirk of tensorflow - the devices are actually named
            # slightly differently from their scopes ... (i.e != "/cpu:0")
            assert variable.device == "/cpu:0" or variable.device == ""

    def test_multi_gpu_shares_variables(self):
        multi_gpu_model = self.get_model(ClassificationModel, self.args)

        single_gpu_args = deepcopy(self.args)
        single_gpu_args["num_gpus"] = 1
        single_gpu_model = self.get_model(ClassificationModel, single_gpu_args)

        multi_gpu_model.train()
        multi_gpu_variables = [x.name for x in multi_gpu_model.model.trainable_weights]

        K.clear_session()
        single_gpu_model.train()
        single_gpu_variables = ["tower_0/" + x.name for x in single_gpu_model.model.trainable_weights]

        assert single_gpu_variables == multi_gpu_variables

    def test_gradient_average(self):
        tensors = [tensorflow.ones([10, 20]) for _ in range(5)]
        average = _get_dense_gradient_average(tensors)
        session = tensorflow.Session()
        numpy.testing.assert_array_equal(session.run(average), session.run(tensors[0]))

    def test_sparse_gradient_average(self):
        tensors = [tensorflow.IndexedSlices(values=tensorflow.ones([5, 20]),
                                            indices=tensorflow.constant([1, 2, 3, 4, 5])) for _ in range(5)]
        average = _get_sparse_gradient_average(tensors)
        session = tensorflow.Session()
        # Unique indices, so the returned tensor should be a weighted average of the respective indices.
        numpy.testing.assert_array_almost_equal(session.run(average.values), session.run(tensorflow.ones([5, 20])))

        tensors = [tensorflow.IndexedSlices(values=tensorflow.ones([5, 20]),
                                            indices=tensorflow.constant([1, 1, 1, 2, 1])) for _ in range(5)]
        average = _get_sparse_gradient_average(tensors)

        # Now we have duplicate indices, so the values for these indices in the 5 tensors we are averaging
        # should be summed prior to being averaged. Here we have 5 tensors x 4 duplicate indices which
        # all have value ones(1, 20), so the first return value should be an array of fours. The second
        # returned value corresponds to the case above. This checks that the slices are being
        # correctly de-duplicated.
        expected_returned_tensor = numpy.concatenate([numpy.ones([1, 20]) * 4., numpy.ones([1, 20])], 0)
        numpy.testing.assert_array_almost_equal(session.run(average.values), expected_returned_tensor)