from keras.layers import merge
from keras.layers.core import Lambda
from .models import DeepQaModel
import tensorflow as tf


def make_parallel(model: DeepQaModel, gpu_count: int):

    """
    :param model: An instance of a DeepQaModel.
    :param gpu_count: The number of GPUs to duplicate the model across.
    :return:
    """

    # Argument to a Lambda layer which will slice our large batches
    # along the batch dimension and return a given slice.
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch.
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU.
                for model_input in model.inputs:
                    # Get the shape of everything apart from the batch,
                    # which will be split across the GPUs.
                    output_shape = tuple(model_input.get_shape().as_list())[1:]
                    slice_layer = Lambda(get_slice,
                                         output_shape=output_shape,
                                         arguments={'idx': i, 'parts': gpu_count})
                    slice_n = slice_layer(model_input)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for output_index in range(len(outputs)):
                    outputs_all[output_index].append(outputs[output_index])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return DeepQaModel(input=model.inputs, output=merged)