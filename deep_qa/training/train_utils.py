from typing import List, Tuple
import tensorflow


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


def average_gradients(tower_gradients: List[List[Tuple]]):
    """
    Given a list of (gradient, variable) pairs from the result of
    a gradient calculation from multiple GPUs, calculate their
    average.
    """
    # Make a map from variables -> [gradients that are not none].
    gradient_map = {}
    for tower in tower_gradients:
        for grad, variable in tower:
            if grad is not None:
                if variable not in gradient_map:
                    gradient_map[variable] = []
                gradient_map[variable].append(grad)
    average_gradient_list = []

    for variable, gradients in gradient_map.items():
        # variable is a tensor.
        # gradients is a list of gradients for this tensor to average.

        # Pick any one of the gradients to see if it is an IndexedSlice.

        first_actual_grad = gradients[0]
        if isinstance(first_actual_grad, tensorflow.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            # gradient with attributes indices and values. To average, we
            # need to concat them individually and then create a new
            # IndexedSlices object.
            indices = []
            values = []
            for grad in gradients:
                indices.append(grad.indices)
                values.append(grad.values)
            all_indices = tensorflow.concat(indices, 0)
            avg_values = tensorflow.concat(values, 0) / len(gradients)
            # Deduplicate across indices.
            values, unique_indicies = _deduplicate_indexed_slices(avg_values, all_indices)
            sparse_grad = tensorflow.IndexedSlices(values,
                                                   unique_indicies,
                                                   dense_shape=first_actual_grad.dense_shape)
            average_gradient_list.append((sparse_grad, variable))
        else:
            # A normal tensor can just do a simple average.
            grads_expanded = []
            for grad in gradients:
                # Add a 0 dimension to the gradients to represent the tower and
                # append on a 'tower' dimension which we will average over.
                grads_expanded.append(tensorflow.expand_dims(grad, 0))

            # Average over the 'tower' dimension.
            grad = tensorflow.concat(grads_expanded, 0)
            mean_grad = tensorflow.reduce_mean(grad, 0)
            average_gradient_list.append((mean_grad, variable))

    assert len(average_gradient_list) == len(gradient_map)
    return average_gradient_list


def _deduplicate_indexed_slices(values, indices):
    """
    Sums `values` associated with any non-unique `indices`.

    Parameters
    ----------
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).

    Returns
    -------
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tensorflow.unique(indices)
    summed_values = tensorflow.unsorted_segment_sum(
            values, new_index_positions,
            tensorflow.shape(unique_indices)[0])
    return summed_values, unique_indices


def slice_batch(batch, num_gpus):

    all_slices = []
    for placeholder in batch:
        # splice placeholder into batches split across the number of gpus specified.
        batch_size = int(placeholder.shape[0] / num_gpus)
        placeholder_slices = []
        for i in range(num_gpus):
            placeholder_slices.append(placeholder[(i * batch_size):((i + 1) * batch_size), ...])
        all_slices.append(placeholder_slices)
    return all_slices
