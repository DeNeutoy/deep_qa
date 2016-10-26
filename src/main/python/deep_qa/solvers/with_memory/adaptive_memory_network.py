
from typing import Any, Dict
from overrides import overrides

import tensorflow as tf

from keras import backend as K
from keras.layers import Reshape, Layer
from keras import initializations
from keras.engine import InputSpec

from .memory_network import MemoryNetworkSolver


class AdapativeStepLayer(Layer):

    def __init__(self, one_minus_epsilon, max_computation, memory_step,
                 initialization='glorot_uniform', name='adaptive_layer', **kwargs):
        self.one_minus_epsilon = one_minus_epsilon
        self.max_computation = max_computation
        self.memory_step = memory_step
        self.init = initializations.get(initialization)
        self.name = name
        super(AdapativeStepLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.input_spec = [InputSpec(shape=input_shape[1])]
        input_dim = input_shape[1][1]
        self.halting_weight = self.init((input_dim, 1), name='{}_halting'.format(self.name))
        self.trainable_weights = [self.halting_weight]

    def call(self, x, mask=None):

        encoded_question, current_memory, encoded_knowledge = x

        batch_mask = tf.cast(tf.ones_like(current_memory[:, 0], name= 'batch_mask'), tf.bool)
        hop_counter = tf.zeros_like(current_memory[:, 0], name='hop_counter')
        halting_accumulator = tf.zeros_like(current_memory[:, 0], name='halting_accumulator')
        halting_accumulator_for_comparison = tf.zeros_like(current_memory[:, 0], name='halting_acc_for_comparision')
        memory_accumulator = tf.zeros_like(current_memory, name='memory_accumulator')
        attended_knowledge_loop_placeholder = tf.zeros_like(current_memory, name='attended_knowledge_placeholder')

        def halting_condition(batch_mask,
                              halting_accumulator,
                              halting_accumulator_for_comparison,
                              hop_counter,
                              encoded_question,
                              current_memory,
                              encoded_knowledge,
                              memory_accumulator,
                              attended_knowledge_placeholder):

            probability_condition = tf.less(halting_accumulator_for_comparison, self.one_minus_epsilon)
            max_computation_condition = tf.less(hop_counter, self.max_computation)

            combined_conditions = tf.logical_and(probability_condition, max_computation_condition)

            return tf.reduce_any(combined_conditions)

        _, _, _, hop_counter, _, _, _, current_memory, attended_knowledge =\
            tf.while_loop(cond=halting_condition, body=self.adaptive_memory_hop,
                          loop_vars=[
                              batch_mask,
                              halting_accumulator,
                              halting_accumulator_for_comparison,
                              hop_counter,
                              encoded_question,
                              current_memory,
                              encoded_knowledge,
                              memory_accumulator,
                              attended_knowledge_loop_placeholder
                          ])

        return [current_memory, attended_knowledge]

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def get_output_shape_for(self, input_shapes):
        return [(input_shapes[1][0], input_shapes[1][1]), (input_shapes[1][0], input_shapes[1][1])]

    def adaptive_memory_hop(self, batch_mask,
                            halting_accumulator,
                            halting_accumulator_for_comparison,
                            hop_counter,
                            encoded_question,
                            previous_memory,
                            encoded_knowledge,
                            memory_accumulator,
                            attended_knowledge):

        current_memory, attended_knowledge = self.memory_step(encoded_question, previous_memory, encoded_knowledge)
        with tf.variable_scope("halting_calculation"):
            halting_probability = tf.squeeze(tf.sigmoid(K.dot(current_memory, self.halting_weight)),1)


        new_batch_mask = tf.logical_and(
            tf.less(halting_accumulator + halting_probability, self.one_minus_epsilon),
            batch_mask)

        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        halting_accumulator += halting_probability * new_float_mask
        halting_accumulator_for_comparison += halting_probability * tf.cast(batch_mask, tf.float32)

        def use_probability():
            masked_halting_probability = tf.expand_dims(halting_probability * new_float_mask, 1)
            accumulated_memory_update = (current_memory * masked_halting_probability) + memory_accumulator
            return accumulated_memory_update

        def use_remainder():
            remainder = tf.expand_dims(1.0 - halting_probability, 1)
            accumulated_memory_update = (current_memory * remainder) + memory_accumulator
            return accumulated_memory_update

        hop_counter += new_float_mask
        counter_condition = tf.less(hop_counter, self.max_computation)
        condition = tf.reduce_any(tf.logical_and(new_batch_mask, counter_condition))

        memory_accumulator = tf.cond(condition, use_probability, use_remainder)

        return_elts = [new_batch_mask,
                halting_accumulator,
                halting_accumulator_for_comparison,
                hop_counter,
                encoded_question,
                current_memory,
                encoded_knowledge,
                memory_accumulator,
                attended_knowledge]
        print("output shapes")
        for elt in return_elts:
            print(elt)

        return return_elts


class AdaptiveMemoryNetworkSolver(MemoryNetworkSolver):

    def __init__(self, params: Dict[str, Any]):

        self.one_minus_epsilon = K.variable(1.0 - params.pop("epsilon", 0.01))
        self.max_computation = K.variable(params.pop("max_computation", 10))
        super(AdaptiveMemoryNetworkSolver, self).__init__(params)

    @overrides
    def _get_memory_network_recurrence(self):

        def adaptive_recurrence(encoded_question, current_memory, encoded_knowledge):
            adaptive_layer = AdapativeStepLayer(self.one_minus_epsilon, self.max_computation, self.memory_step)
            return adaptive_layer([encoded_question, current_memory, encoded_knowledge])

        return adaptive_recurrence










