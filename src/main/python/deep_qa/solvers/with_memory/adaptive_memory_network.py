

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

        self.one_minus_epsilon = 1.0 - params.pop("epsilon", 0.01)
        self.max_computation = params.pop("max_computation", 10)

        super(AdaptiveMemoryNetworkSolver, self).__init__(params)

    @overrides
    def _build_model(self):


        question_input_layer, question_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_question_shape(), name_prefix="sentence")
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(
            input_shape=self._get_background_shape(), name_prefix="background")

        # Step 3: Encode the two embedded inputs using the sentence encoder.
        question_encoder = self._get_sentence_encoder()
        encoded_question = question_encoder(question_embedding)  # (samples, word_dim)

        # Knowledge encoder will have the same encoder running on a higher order tensor.
        # i.e., question_encoder: (samples, num_words, word_dim) -> (samples, word_dim)
        # and knowledge_encoder: (samples, knowledge_len, num_words, word_dim) ->
        #                       (samples, knowledge_len, word_dim)
        # TimeDistributed generally loops over the second dimension.
        knowledge_encoder = self._get_knowledge_encoder()
        encoded_knowledge = knowledge_encoder(knowledge_embedding)  # (samples, knowledge_len, word_dim)

        self.knowledge_combiner = self._get_knowledge_combiner(0)
        self.knowledge_axis = self._get_knowledge_axis()

        current_memory = encoded_question

        halting_condition =
        batch_mask =
        hop_counter =
        halting_accumulator =
        halting_accumulator_for_comparison =
        memory_accumulator =

        _, _, _, hop_counter, _, _, _, current_memory, attended_knowledge = tf.while_loop(cond=halting_condition,
                                   body=self.memory_hop,
                                   loop_vars=[batch_mask,
                                              halting_accumulator,
                                              halting_accumulator_for_comparison,
                                              hop_counter,
                                              encoded_question,
                                              current_memory,
                                              encoded_knowledge,
                                              memory_accumulator])

        # Step 5: Finally, run the sentence encoding, the current memory, and the attended
        # background knowledge through an entailment model to get a final true/false score.
        entailment_input = merge([encoded_question, current_memory, attended_knowledge],
                                 mode='concat',
                                 concat_axis=self.knowledge_axis,
                                 name='concat_entailment_inputs')
        combined_input = self._get_entailment_input_combiner()(entailment_input)
        extra_entailment_inputs, entailment_output = self._get_entailment_output(combined_input)

        # Step 6: Define the model, and return it. The model will be compiled and trained by the
        # calling method.
        input_layers = [question_input_layer, knowledge_input_layer]
        input_layers.extend(extra_entailment_inputs)
        memory_network = Model(input=input_layers, output=entailment_output)
        return memory_network

    def memory_hop(self, batch_mask,
                   halting_accumulator,
                   halting_accumulator_for_comparison,
                   hop_counter,
                   encoded_question,
                   previous_memory,
                   encoded_knowledge,
                   memory_accumulator,
                   attended_knowledge):


        current_memory, attended_knowledge = self.internal_memory_update(encoded_question, previous_memory, encoded_knowledge)

        with tf.variable_scope("halting_calculation"):

            halting_probability = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear(current_memory, 1, bias=True)))


        new_batch_mask = tf.logical_and(tf.less(halting_accumulator + halting_probability,self.one_minus_epsilon),batch_mask)
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
        counter_condition = tf.less(hop_counter,self.batch_max_computation)
        condition = tf.reduce_any(tf.logical_and(new_batch_mask,counter_condition))

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

    def internal_memory_update(self, encoded_question, current_memory, encoded_knowledge):

        merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=self.knowledge_axis),
                                                       K.expand_dims(layer_outs[1], dim=self.knowledge_axis),
                                                       layer_outs[2]],
                                                      axis=self.knowledge_axis)

        merged_shape = self._get_merged_background_shape()
        merged_encoded_rep = merge([encoded_question, current_memory, encoded_knowledge],
                                   mode=merge_mode,
                                   output_shape=merged_shape,
                                   name='concat_memory_and_question_with_background_%d' % i)

        # Regularize it
        regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
        knowledge_selector = self._get_knowledge_selector(i)
        attention_weights = knowledge_selector(regularized_merged_rep)
        # Defining weighted average as a custom merge mode. Takes two inputs: data and weights
        # ndim of weights is one less than data.

        # We now combine the knowledge and the weights using the knowledge combiner. In order to
        # make it easy to TimeDistribute these Layers for use with different solvers,
        # we prepend the attention mask onto the background knowledge.
        # Note that this merge assumes that the word_dim is always after the background_length dimension.
        # (samples, knowledge_length, word_dim), (samples, knowledge_length)
        #                                    => (samples, knowledge_length, 1 + word_dim)

        concat_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=self.knowledge_axis + 1),
                                                        layer_outs[1]],
                                                       axis=self.knowledge_axis + 1)
        concat_shape = self._get_concat_background_shape()

        combined_background_with_attention = merge([attention_weights, encoded_knowledge],
                                                   mode=concat_mode,
                                                   output_shape=concat_shape,
                                                   name='concat_attention_with_background_%d' % i)

        attended_knowledge = self.knowledge_combiner(combined_background_with_attention)

        # To make this easier to TimeDistribute, we're going to concatenate the current memory
        # with the attended knowledge, and use that as the input to the memory updater, instead
        # of just passing a list.
        # We going from two inputs of (batch_size, encoding_dim) to one input of (batch_size,
        # encoding_dim * 2).
        updater_input = merge([encoded_question, current_memory, attended_knowledge],
                              mode='concat',
                              concat_axis=self.knowledge_axis,
                              name='concat_current_memory_with_background_%d' % i)
        memory_updater = self._get_memory_updater(i)
        current_memory = memory_updater(updater_input)

        return current_memory, attended_knowledge






