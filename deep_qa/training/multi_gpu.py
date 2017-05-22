import tensorflow
import keras.backend as K
from ..common.params import Params

from .train_utils import _average_gradients


def pin_variable_device_scope(device, variable_device="/cpu:0"):
    """
    Tensorflow device scopes can take functions which give a device
    for a given op in the graph. Here, we use the device that is
    passed to the scope *unless* the operation which is being created
    in the graph is a Variable creation op; in this case, we place it
    on the cpu.
    """
    def _assign(graph_op):
        node_def = graph_op if isinstance(graph_op, tensorflow.NodeDef) else graph_op.node_def
        if node_def.op in ['Variable', 'VariableV2']:
            return variable_device
        else:
            return device
    return _assign


class MultiGpuModel(object):

    def __init__(self, params: Params):

        self.model = params.pop("model")
        self.gpu_count = params.pop("gpu_count")

    def build_model(self):

        tower_models = []
        tower_gradients = []
        global_step = tensorflow.train.get_or_create_global_step()
        train_loss = tensorflow.get_variable('train_loss', [],
            initializer=tensorflow.constant_initializer(0.0), trainable=False)

        # Place a copy of the model on each GPU, each getting a slice of the batch.
        for gpu_index in range(self.gpu_count):
            with tensorflow.device(pin_variable_device_scope('/gpu:%d' % gpu_index)):
                with tensorflow.name_scope('tower_%d' % gpu_index):

                    # TODO: check we don't need to compile every time.
                    loss = self.model.total_loss
                    grads = self.model.optimiser.compute_gradients(loss)
                    tower_gradients.append(grads)
                    train_loss += loss

        grads = _average_gradients(tower_gradients)
        train_operation = self.model.optimiser.apply_gradients(grads,
                                                        self.model._collected_trainable_weights,
                                                        global_step=global_step)

        updates = self.model.updates + [train_operation]
        inputs = self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights

        inputs = []
        updates = []
        for model in tower_models:
            model_inputs = (model._feed_inputs + model._feed_targets +
                            model._feed_sample_weights)
            inputs.extend(model_inputs)
            updates.extend(model.updates)
        # Just check any one, as we just made copies of them.
        if tower_models[0].uses_learning_phase and \
                not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]

        # Add the multi-gpu update operation
        updates += [train_operation]

        # Gets loss and metrics. Updates weights at each call.

        self.model.train_function = K.Function(inputs, [train_loss] + self.model.metrics_tensors, updates=updates)

    def fit_parallel(self, inputs, outputs):

    # TODO: Eventually make single gpu models train like this.
    def make_parallel_train_function(self):
