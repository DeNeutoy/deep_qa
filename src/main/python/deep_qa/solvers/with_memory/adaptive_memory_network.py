
from typing import Any, Dict
from overrides import overrides

import tensorflow as tf

from keras import backend as K

from .memory_network import MemoryNetworkSolver


class AdaptiveMemoryNetworkSolver(MemoryNetworkSolver):

    def __init__(self, params: Dict[str, Any]):

        self.one_minus_epsilon = K.variable(1.0 - params.pop("epsilon", 0.01))
        self.max_computation = K.variable(params.pop("max_computation", 10))

        super(AdaptiveMemoryNetworkSolver, self).__init__(params)

    @overrides
    def _get_memory_network_recurrence(self):

        def adaptive_recurrence(encoded_question, current_memory, encoded_knowledge):

            batch_mask = tf.squeeze(tf.cast(tf.ones_like(current_memory[:, 0], name= 'batch_mask'), tf.bool))
            hop_counter = tf.squeeze(tf.zeros_like(current_memory[:, 0], name='hop_counter'))
            halting_accumulator = tf.squeeze(tf.zeros_like(current_memory[:, 0], name='halting_accumulator'))
            halting_accumulator_for_comparison = tf.squeeze(tf.zeros_like(current_memory[:, 0], name='halting_acc_for_comparision'))
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
            return current_memory, attended_knowledge

        return adaptive_recurrence

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
            halting_probability = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear(current_memory, 1, bias=True)))

        print("halting prob:", K.shape(halting_probability))
        new_batch_mask = tf.logical_and(
            tf.less(halting_accumulator + halting_probability,
                    self.one_minus_epsilon), batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        halting_accumulator += halting_probability * new_float_mask
        halting_accumulator_for_comparison += halting_probability * tf.cast(batch_mask, tf.float32)

        def use_probability():
            masked_halting_probability = tf.tile(tf.expand_dims(halting_probability * new_float_mask, 1), [1, 5])
            print("expanded mask", K.shape(masked_halting_probability))
            print("memory accumulator", K.shape(memory_accumulator))
            print(current_memory)
            accumulated_memory_update = (current_memory * masked_halting_probability) + memory_accumulator
            print("accumulated memory update", K.shape(accumulated_memory_update))
            return accumulated_memory_update

        def use_remainder():
            remainder = tf.expand_dims(1.0 - halting_probability, 1)
            accumulated_memory_update = (current_memory * remainder) + memory_accumulator
            return accumulated_memory_update

        hop_counter += new_float_mask
        counter_condition = tf.less(hop_counter, self.max_computation)
        condition = tf.reduce_any(tf.logical_and(new_batch_mask, counter_condition))

        #memory_accumulator = tf.cond(condition, use_probability, use_remainder)

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






