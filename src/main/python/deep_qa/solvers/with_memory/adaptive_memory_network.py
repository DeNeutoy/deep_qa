

from copy import deepcopy
from typing import Any, Dict, List
from overrides import overrides

import tensorflow as tf

from keras import backend as K
from keras.layers import Dropout, merge
from keras.models import Model

from .memory_network import MemoryNetworkSolver
from ...common.params import get_choice_with_default
from ...data.dataset import Dataset, TextDataset
from ...data.instances.true_false_instance import TrueFalseInstance
from ...layers.knowledge_selectors import selectors
from ...layers.memory_updaters import updaters
from ...layers.entailment_models import entailment_models, entailment_input_combiners
from ...layers.knowledge_combiners import knowledge_combiners
from ...layers.knowledge_encoders import knowledge_encoders



class AdaptiveMemoryNetworkSolver(MemoryNetworkSolver):

    def __init__(self, params: Dict[str, Any]):

        self.one_minus_epsilon = K.variable(1.0 - params.pop("epsilon", 0.01))
        self.max_computation = K.variable(params.pop("max_computation", 10))

        super(AdaptiveMemoryNetworkSolver, self).__init__(params)

    @overrides
    def _get_memory_network_recurrence(self):

        def adaptive_memory_network_recurrence(encoded_question, current_memory, encoded_knowledge):

            batch_mask = K.cast(K.ones_like(current_memory[:, 0]), 'bool')
            hop_counter = K.zeros_like(current_memory[:, 0])
            halting_accumulator = K.zeros_like(current_memory[:, 0])
            halting_accumulator_for_comparison = K.zeros_like(current_memory[:, 0])
            memory_accumulator = K.zeros_like(current_memory)
            attended_knowledge_loop_placeholder = K.zeros_like(current_memory)

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

            return adaptive_memory_network_recurrence

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

        new_batch_mask = tf.logical_and(
            tf.less(halting_accumulator + halting_probability,
                    self.one_minus_epsilon), batch_mask)
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

        return (new_batch_mask,
                halting_accumulator,
                halting_accumulator_for_comparison,
                hop_counter,
                encoded_question,
                current_memory,
                encoded_knowledge,
                memory_accumulator,
                attended_knowledge)








