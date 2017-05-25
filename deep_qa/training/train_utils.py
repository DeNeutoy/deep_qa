
import tensorflow
import numpy

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


def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tensorflow.unique(indices)
    summed_values = tensorflow.unsorted_segment_sum(
        values, new_index_positions,
        tensorflow.shape(unique_indices)[0])
    return (summed_values, unique_indices)


def _average_gradients(tower_grads):
    # tower_grads = [
    #       [(grad1, var1), ..(gradn, varn)] for tower 1
    #       [(grad1, var1), ..(gradn, varn)] for tower 2,  ...
    #
    # calculate average gradient for each shared variable across all GPUs

    # make a map var -> [grads that are not none]
    grad_map = {}
    for tower in tower_grads:
        for g, v in tower:
            if g is not None:
                if v not in grad_map:
                    grad_map[v] = []
                grad_map[v].append(g)

    average_grads = []

    for v, grads in grad_map.items():
        # v = tensor
        # grads = list of gradients for this tensor to average

        g0 = grads[0]
        if isinstance(g0, tensorflow.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g in grads:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tensorflow.concat(indices, 0)
            avg_values = tensorflow.concat(values, 0) / len(grads)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tensorflow.IndexedSlices(av, ai, dense_shape=g0.dense_shape)
            average_grads.append((grad, v))
        else:
            # a normal tensor can just do a simple average
            grads_expanded = []
            for g in grads:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tensorflow.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over
                grads_expanded.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tensorflow.concat(grads_expanded, 0)
            grad = tensorflow.reduce_mean(grad, 0)

            average_grads.append((grad, v))

    assert len(average_grads) == len(grad_map)

    return average_grads


def _clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    '''
    wrapper around tensorflow.clip_by_global_norm that also does summary ops of norms
    compute norms
    '''
    # 0 compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tensorflow.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name
        summary_ops.append(tensorflow.summary.scalar(name, ns))

    # clip
    clipped_t_list, tensorflow_norm = tensorflow.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tensorflow.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name
        summary_ops.append(tensorflow.summary.scalar(name, ns))

    summary_ops.append(tensorflow.summary.scalar(norm_name, tensorflow_norm))

    return clipped_t_list, tensorflow_norm, summary_ops


def _clip_grads(grads, all_clip_norm_val, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val * tensorflow.minimum((global_step + 1) / 100.0, 1.0)
        clipped_tensors, g_norm, so = _clip_by_global_norm_summary(
            grad_tensors, scaled_val, name, vv)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    if all_clip_norm_val is not None:
        # get the grads that aren't None
        grads_not_none = [(g, v) for g, v in grads if g is not None]
        grads_none = [(g, v) for g, v in grads if g is None]
        ret, summary_ops = _clip_norms(
            grads_not_none, all_clip_norm_val, 'norm_grad')
        ret.extend(grads_none)

    assert len(ret) == len(grads)

    return ret, summary_ops


def slice_batch_original(batch, n_gpus):
    # batch = either X or y with batch dimension batch_size * n_gpus
    #   slices keys in X and y to multiple GPU slices
    # return: {varname: [slice1, slice2, .., sliceN], ..}
    ret = {}
    for key, v in batch.items():
        # splice v into n_gpus
        bs = int(v.shape[0] / n_gpus)
        v_slice = []
        for k in range(n_gpus):
            v_slice.append(v[(k * bs):((k + 1) * bs), ...])
        ret[key] = v_slice
    return ret


def create_batches(inputs, labels, batch_size):

    return_array = []
    for i in range(0, len(inputs[0]) - batch_size, batch_size):

        return_array.append((([inputs[i][i*batch_size: (i+1) * batch_size] for i in range(len(inputs))],
                             [labels[i*batch_size: (i+1) * batch_size] for i in range(len(labels))])))

    return return_array


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


def _scale_grads(grads, grad_scale_spec):
    # grad_scale_spec:
    #  [['variable_name', fac], ..]
    ret = []
    for g, v in grads:
        vname = v.name
        for name_spec, fac_spec in grad_scale_spec:
            if name_spec in vname:
                print("SCALING {0} gradient by {1}".format(
                    vname, fac_spec))
                if isinstance(g, tensorflow.IndexedSlices):
                    g.values = g.values * fac_spec
                else:
                    g = g * fac_spec
        ret.append([g, v])
    return ret
