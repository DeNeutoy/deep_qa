from overrides import overrides
import logging
import six

from keras.models import Model, Sequential
from keras import losses
from keras import metrics as metrics_module
from keras.engine.training import _weighted_masked_objective, _collect_metrics, _masked_objective
import keras.backend as K
import tensorflow

from ..common.params import Params, ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DeepQaModel(Model):
    """
    This is a Model that adds functionality to Keras' ``Model`` class. In
    particular, we use tensorflow optimisers directly in order to make use
    of sparse gradient updates, which Keras does not handle. Additionally,
    we provide some nicer summary functions which include mask information.
    We are overriding key components of Keras here and you should probably
    have a pretty good grip on the internals of Keras before you change
    stuff below, as there could be unexpected consequences.
    """

    # TODO(Mark): Tensorflow optimisers are not compatible with Keras' LearningRateScheduler.
    def __init__(self, *args, **kwargs):
        super(DeepQaModel, self).__init__(*args, **kwargs)

    # We want to add a few things to the summary that's printed out by Keras.  Unfortunately, Keras
    # makes that very difficult.  We have to copy large portions of code in order to make this
    # work, because `print_summary()` is in `keras.utils.layer_utils`, instead of a member on
    # `Container`...
    @overrides
    def summary(self, show_masks=False, **kwargs):
        if show_masks:
            self._summary_with_mask_info()
        else:
            self._keras_summary(**kwargs)

    def _keras_summary(self):
        super(DeepQaModel, self).summary()

    def _summary_with_mask_info(self):
        flattened_layers = getattr(self, 'flattened_layers', self.layers)
        print_summary_with_masking(flattened_layers, getattr(self, 'container_nodes', None))

    @overrides
    def _make_train_function(self):
        # pylint: disable=attribute-defined-outside-init
        """
        We override this method so that we can use tensorflow optimisers directly.
        This is desirable as tensorflow handles gradients of sparse tensors efficiently.
        """
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.train_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]

            # Here we override Keras to use tensorflow optimizers directly.
            self.global_step = K.variable(0., name='global_step')
            gradients = tensorflow.gradients(self.total_loss, self._collected_trainable_weights)
            if self.gradient_clipping is not None:
                # Don't pop from the gradient clipping dict here as
                # if we call fit more than once we need it to still be there.
                clip_type = self.gradient_clipping.get("type")
                clip_value = self.gradient_clipping.get("value")
                if clip_type == 'clip_by_norm':
                    gradients, _ = tensorflow.clip_by_global_norm(gradients, clip_value)
                elif clip_type == 'clip_by_value':
                    gradients = [tensorflow.clip_by_value(x, -clip_value, clip_value) for x in gradients]
                else:
                    raise ConfigurationError("{} is not a supported type of gradient clipping.".format(clip_type))

            zipped_grads_with_weights = zip(gradients, self._collected_trainable_weights)
            # pylint: disable=no-member
            training_updates = self.optimizer.apply_gradients(zipped_grads_with_weights,
                                                              global_step=self.global_step)
            # pylint: enable=no-member
            updates = self.updates + [training_updates]
            # Gets loss and metrics. Updates weights at each call.
            self.train_function = K.Function(inputs, [self.total_loss] + self.metrics_tensors, updates=updates)

    @overrides
    def compile(self, params: Params):
        """Configures the model for training.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: str (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
            metrics: list of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions
                of different model outputs.
                If a list, it is expected to have a 1:1 mapping
                to the model's outputs. If a tensor, it is expected to map
                output names (strings) to scalar coefficients.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to `"temporal"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.

        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """
        # TODO: change this to get the actual instance here rather than in Trainer
        self.optimizer = params.pop('optimizer')
        self.gradient_clipping = params.pop('gradient_clipping', None).as_dict()
        self.loss = loss = params.pop('loss', {})
        self.sample_weight_mode = sample_weight_mode = params.pop('sample_weight_mode', None)
        self.loss_weights = loss_weights = params.pop('loss_weights', None)
        metrics = params.pop('metrics', None)


        # Get the loss functions for all the outputs of the model.
        # TODO: see if this has to be an attribute, it's only used in this function in model.
        self.loss_functions = self.get_loss_functions(loss)
        weighted_losses = [_weighted_masked_objective(fn) for fn in self.loss_functions]

        skip_indices = []
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss_fns = []
        for i in range(len(weighted_losses)):
            if weighted_losses[i] is None:
                skip_indices.append(i)
            else:
                self._feed_outputs.append(self.outputs[i])
                self._feed_output_names.append(self.output_names[i])
                self._feed_output_shapes.append(self.internal_output_shapes[i])
                self._feed_loss_fns.append(self.loss_functions[i])

        # Prepare output masks.
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        if not isinstance(masks, list):
            masks = [masks]

        # Prepare loss weights.
        loss_weights_list = self.get_loss_weights(loss_weights)

        # Prepare sample weights.
        sample_weights, self.sample_weight_modes = self.get_sample_weights(sample_weight_mode, skip_indices)
        self._feed_sample_weight_modes = []
        for i in range(len(self.outputs)):
            if i not in skip_indices:
                self._feed_sample_weight_modes.append(self.sample_weight_modes[i])

        # Prepare targets of model.
        self.targets = []
        self._feed_targets = []
        for i in range(len(self.outputs)):
            if i in skip_indices:
                self.targets.append(None)
            else:
                shape = self.internal_output_shapes[i]
                name = self.output_names[i]
                target = K.placeholder(ndim=len(shape),
                                       name=name + '_target',
                                       sparse=K.is_sparse(self.outputs[i]),
                                       dtype=K.dtype(self.outputs[i]))
                self.targets.append(target)
                self._feed_targets.append(target)

        # Prepare metrics.
        self.metrics = metrics
        self.metrics_names = ['loss']
        self.metrics_tensors = []

        # Compute total loss.
        total_loss = None
        for i in range(len(self.outputs)):
            if i in skip_indices:
                continue
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            weighted_loss = weighted_losses[i]
            sample_weight = sample_weights[i]
            mask = masks[i]
            loss_weight = loss_weights_list[i]
            output_loss = weighted_loss(y_true, y_pred,
                                        sample_weight, mask)
            if len(self.outputs) > 1:
                self.metrics_tensors.append(output_loss)
                self.metrics_names.append(self.output_names[i] + '_loss')
            if total_loss is None:
                total_loss = loss_weight * output_loss
            else:
                total_loss += loss_weight * output_loss
        if total_loss is None:
            if not self.losses:
                raise RuntimeError('The model cannot be compiled '
                                   'because it has no loss to optimize.')
            else:
                total_loss = 0.

        # Add regularization penalties
        # and other layer-specific losses.
        for loss_tensor in self.losses:
            total_loss += loss_tensor

        # List of same size as output_names.
        # contains tuples (metrics for output, names of metrics).
        nested_metrics = _collect_metrics(metrics, self.output_names)

        def append_metric(layer_num, metric_name, metric_tensor):
            """Helper function used in loop below."""
            if len(self.output_names) > 1:
                metric_name = self.output_layers[layer_num].name + '_' + metric_name
            self.metrics_names.append(metric_name)
            self.metrics_tensors.append(metric_tensor)

        for i in range(len(self.outputs)):
            if i in skip_indices:
                continue
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            output_metrics = nested_metrics[i]
            for metric in output_metrics:
                if metric == 'accuracy' or metric == 'acc':
                    # custom handling of accuracy
                    # (because of class mode duality)
                    output_shape = self.internal_output_shapes[i]
                    acc_fn = None
                    if (output_shape[-1] == 1 or
                       self.loss_functions[i] == losses.binary_crossentropy):
                        # case: binary accuracy
                        acc_fn = metrics_module.binary_accuracy
                    elif self.loss_functions[i] == losses.sparse_categorical_crossentropy:
                        # case: categorical accuracy with sparse targets
                        acc_fn = metrics_module.sparse_categorical_accuracy
                    else:
                        acc_fn = metrics_module.categorical_accuracy

                    masked_fn = _masked_objective(acc_fn)
                    append_metric(i, 'acc', masked_fn(y_true, y_pred, mask=masks[i]))
                else:
                    metric_fn = metrics_module.get(metric)
                    masked_metric_fn = _masked_objective(metric_fn)
                    metric_result = masked_metric_fn(y_true, y_pred, mask=masks[i])
                    metric_result = {
                        metric_fn.__name__: metric_result
                    }
                    for name, tensor in six.iteritems(metric_result):
                        append_metric(i, name, tensor)

        # Prepare gradient updates and state updates.
        self.total_loss = total_loss
        self.sample_weights = sample_weights
        self._feed_sample_weights = []
        for i in range(len(self.sample_weights)):
            if i not in skip_indices:
                self._feed_sample_weights.append(sample_weights[i])

        # Functions for train, test and predict will
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Collected trainable weights and sort them deterministically.
        trainable_weights = self.trainable_weights
        # Sort weights by name.
        if trainable_weights:
            if K.backend() == 'theano':
                trainable_weights.sort(key=lambda x: x.name if x.name else x.auto_name)
            else:
                trainable_weights.sort(key=lambda x: x.name)
        self._collected_trainable_weights = trainable_weights


    def get_loss_functions(self, loss):

        """
        STEP 1 of Model.compile. Build the loss functions for each of the outputs
        of the model.
        """

        # Prepare loss functions.
        if isinstance(loss, dict):
            for name in loss:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_functions = []
            for name in self.output_names:
                if name not in loss:
                    logger.warning('Output "' + name +
                                  '" missing from loss dictionary. '
                                  'We assume this was done on purpose, '
                                  'and we will not be expecting '
                                  'any data to be passed to "' + name +
                                  '" during training.')
                loss_functions.append(losses.get(loss.get(name)))
        elif isinstance(loss, list):
            if len(loss) != len(self.outputs):
                raise ValueError('When passing a list as loss, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss=' +
                                 str(loss))
            loss_functions = [losses.get(l) for l in loss]
        else:
            loss_function = losses.get(loss)
            loss_functions = [loss_function for _ in range(len(self.outputs))]

        return loss_functions

    def get_loss_weights(self, loss_weights):

        """
        STEP 2 of Model.compile: if we are weighting the losses, extract them from
        the variety of formats keras supports for this.
        """

        if loss_weights is None:
            loss_weights_list = [1. for _ in range(len(self.outputs))]
        elif isinstance(loss_weights, dict):
            for name in loss_weights:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss_weights '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_weights_list = []
            for name in self.output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif isinstance(loss_weights, list):
            if len(loss_weights) != len(self.outputs):
                raise ValueError('When passing a list as loss_weights, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss_weights=' +
                                 str(loss_weights))
            loss_weights_list = loss_weights
        else:
            raise TypeError('Could not interpret loss_weights argument: ' +
                            str(loss_weights) +
                            ' - expected a list of dicts.')

        return loss_weights_list

    def get_sample_weights(self, sample_weight_mode, skip_indices):

        """
        STEP #: Get the sample weights....
        """

        sample_weights = []
        sample_weight_modes = []
        if isinstance(sample_weight_mode, dict):
            for name in sample_weight_mode:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in '
                                     'sample_weight_mode dictionary: "' +
                                     name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            for i, name in enumerate(self.output_names):
                if i in skip_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    if name not in sample_weight_mode:
                        raise ValueError('Output "' + name +
                                         '" missing from sample_weight_modes '
                                         'dictionary')
                    if sample_weight_mode.get(name) == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif isinstance(sample_weight_mode, list):
            if len(sample_weight_mode) != len(self.outputs):
                raise ValueError('When passing a list as sample_weight_mode, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed '
                                 'sample_weight_mode=' +
                                 str(sample_weight_mode))
            for i in range(len(self.output_names)):
                if i in skip_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    mode = sample_weight_mode[i]
                    name = self.output_names[i]
                    if mode == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            for i, name in enumerate(self.output_names):
                if i in skip_indices:
                    sample_weight_modes.append(None)
                    sample_weights.append(None)
                else:
                    if sample_weight_mode == 'temporal':
                        sample_weights.append(
                            K.placeholder(ndim=2,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append('temporal')
                    else:
                        sample_weights.append(
                            K.placeholder(ndim=1,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append(None)

        return sample_weights, sample_weight_modes

    def _make_test_function(self):

        """
        Only difference here: We don't use **kwargs in the K.function call,
        as they are only supported by Theano anyway.
        """

        if not hasattr(self, 'test_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.test_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            # Return loss and metrics, no gradient updates.
            # Does update the network states.
            self.test_function = K.function(inputs,
                                            [self.total_loss] + self.metrics_tensors,
                                            updates=self.state_updates)





def print_summary_with_masking(layers, relevant_nodes=None):
    line_length = 150
    positions = [40, 60, 68, 98, 124, 150]
    headers = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to', 'Input mask', 'Output mask']

    print('_' * line_length)
    print_row(headers, positions)
    print('=' * line_length)

    for i, layer in enumerate(layers):
        print_layer_summary(layer, relevant_nodes, positions)
        if i == len(layers) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)

    print('Total params: %s' % count_total_params(layers))
    print('_' * line_length)


def print_row(fields, positions):
    line = ''
    for field, position in zip(fields, positions):
        line += str(field)
        line = line[:position - 1]
        line += ' ' * (position - len(line))
    print(line)


def print_layer_summary(layer, relevant_nodes, positions):
    try:
        output_shape = layer.output_shape
    except Exception:  # pylint: disable=broad-except
        output_shape = 'multiple'
    connections = []
    input_masks = []
    output_masks = []
    for node_index, node in enumerate(layer.inbound_nodes):
        input_mask = layer.get_input_mask_at(node_index)
        if isinstance(input_mask, list):
            input_masks.extend(input_mask)
        else:
            input_masks.append(input_mask)
        output_masks.append(layer.get_output_mask_at(node_index))
        if relevant_nodes:
            node_key = layer.name + '_ib-' + str(node_index)
            if node_key not in relevant_nodes:
                # node is node part of the current network
                continue
        for i in range(len(node.inbound_layers)):
            inbound_layer = node.inbound_layers[i].name
            inbound_node_index = str(node.node_indices[i])
            inbound_tensor_index = str(node.tensor_indices[i])
            connections.append(inbound_layer + '[' + inbound_node_index + '][' + inbound_tensor_index + ']')

    name = layer.name
    cls_name = layer.__class__.__name__
    first_connection = '' if not connections else connections[0]
    first_input_mask = '' if not input_masks else input_masks[0]
    first_output_mask = '' if not output_masks else output_masks[0]
    fields = [
            name + ' (' + cls_name + ')',
            output_shape,
            layer.count_params(),
            first_connection,
            first_input_mask,
            first_output_mask,
            ]
    print_row(fields, positions)
    rows_needed = max(len(connections), len(output_masks), len(input_masks))
    for i in range(1, rows_needed):
        connection = '' if i >= len(connections) else connections[i]
        input_mask = '' if i >= len(input_masks) else input_masks[i]
        output_mask = '' if i >= len(output_masks) else output_masks[i]
        fields = ['', '', '', connection, input_mask, output_mask]
        print_row(fields, positions)


def count_total_params(layers, layer_set=None):
    if layer_set is None:
        layer_set = set()
    total_params = 0
    for layer in layers:
        if layer in layer_set:
            continue
        layer_set.add(layer)
        if isinstance(layer, Model) or isinstance(layer, Sequential):
            total_params += count_total_params(layer.layers, layer_set)
        else:
            total_params += layer.count_params()
    return total_params


def _slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `_slice_arrays(x, indices)`

    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    # Returns
        A slice of the array(s).
    """
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]
