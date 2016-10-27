from typing import Any, Dict
from overrides import overrides

import tensorflow as tf

from keras import backend as K
from keras.layers import Layer
from keras import initializations
from keras.regularizers import l1
from keras.engine import InputSpec
from .memory_network import MemoryNetworkSolver


class AdaptiveMemoryNetworkSolver(MemoryNetworkSolver):
    '''
    This solver uses an Adaptive number of memory steps which is learnt during training.
    Other than this, it is identical to the standard MemoryNetworkSolver.
    '''
    def __init__(self, params: Dict[str, Any]):

        self.one_minus_epsilon = K.variable(1.0 - params.pop("epsilon", 0.01))
        self.max_computation = K.variable(params.pop("max_computation", 10))
        self.ponder_cost_param = params.pop("ponder_cost_param", 0.05)
        super(AdaptiveMemoryNetworkSolver, self).__init__(params)

    @overrides
    def _get_memory_network_recurrence(self):

        # Instead of running for a fixed number of steps
        def adaptive_recurrence(encoded_question, current_memory, encoded_knowledge):
            adaptive_layer = AdaptiveStep(self.one_minus_epsilon, self.max_computation,
                                          self.memory_step, self.ponder_cost_param)
            return adaptive_layer([encoded_question, current_memory, encoded_knowledge])

        return adaptive_recurrence


class AdaptiveStep(Layer):
    '''
    This layer implements a single step of the halting component of the Adaptive Computation Time algorithm,
    generalised so that it can be applied to any arbitrary function. Here, that function is a single memory network
    step. This can be seen as a differentiable while loop, where the halting condition is an accumulated
    'halting probability' which is computed at every memory network step.

    The main machinery implemented here is to deal with doing this process with batched inputs.
    There is a subtlety here regarding the batch_size, as clearly we will have samples halting
    at different points in the batch. This is dealt with using logical masks to protect accumulated
    probabilities, states and outputs from a timestep t's contribution if they have already reached
    1-es at a timestep s < t.
    '''
    def __init__(self, one_minus_epsilon, max_computation, memory_step, ponder_cost_param,
                 initialization='glorot_uniform', name='adaptive_layer', **kwargs):
        self.one_minus_epsilon = one_minus_epsilon
        self.max_computation = max_computation
        self.ponder_cost_param = ponder_cost_param
        self.memory_step = memory_step
        self.init = initializations.get(initialization)
        self.name = name
        super(AdaptiveStep, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        The only weight that this layer requires is used in a simple dot product with the current_memory
        to generate the halting_probability. We define the weight shape with the 2nd input to this
        layer, as this is the memory representation, which will dictate the required size.
        '''
        self.input_spec = [InputSpec(shape=input_shape[1])]
        input_dim = input_shape[1][1]
        self.halting_weight = self.init((input_dim, 1), name='{}_halting_weight'.format(self.name))
        self.trainable_weights = [self.halting_weight]

    def call(self, x, mask=None):
        encoded_question, current_memory, encoded_knowledge = x

        # This is a boolean mask, holding whether a particular sample has halted.
        batch_mask = tf.cast(tf.ones_like(current_memory[:, 0], name= 'batch_mask'), tf.bool)
        # This counts the number of memory steps per sample.
        hop_counter = tf.zeros_like(current_memory[:, 0], name='hop_counter')
        # This accumulates the halting probabilities.
        halting_accumulator = tf.zeros_like(current_memory[:, 0], name='halting_accumulator')
        # This also accumulates the halting probabilities, with the difference being that if an outputed probability
        # causes a particular sample to go over 1 - epsilon, this accumulates that value, but the halting_accumulator
        # does not. This variable is _only_ used in the halting condition of the loop.
        halting_accumulator_for_comparison = tf.zeros_like(current_memory[:, 0], name='halting_acc_for_comparision')
        # This accumulates the weighted memory vectors at each memory step. The memory is weighted by the
        # halting probability and added to this accumulator.
        memory_accumulator = tf.zeros_like(current_memory, name='memory_accumulator')
        # We need the attended_knowledge from the last memory network step, so we create a dummy variable to
        # input to the while_loop, as tensorflow requires the input signature to match the output signature.
        attended_knowledge_loop_placeholder = tf.zeros_like(current_memory, name='attended_knowledge_placeholder')

        ponder_cost = l1(self.ponder_cost_param)
        ponder_cost.set_param(hop_counter)
        self.regularizers.append(ponder_cost)

        # Tensorflow requires that we use all of the variables used in the tf.while_loop as inputs to the
        # condition for halting the loop, even though we only actually make use of two of them.
        def halting_condition(batch_mask,
                              halting_accumulator,
                              halting_accumulator_for_comparison,
                              hop_counter,
                              encoded_question,
                              current_memory,
                              encoded_knowledge,
                              memory_accumulator,
                              attended_knowledge_placeholder):
            # This condition checks the batch elementwise to see if any of the accumulated halting probabilities have
            # gone over one_minus_epsilon in the previous iteration.
            probability_condition = tf.less(halting_accumulator_for_comparison, self.one_minus_epsilon)

            # This condition checks the batch elementwise to see if any have taken more steps than the max allowed.
            max_computation_condition = tf.less(hop_counter, self.max_computation)
            # We only stop if both of the above conditions are true....
            combined_conditions = tf.logical_and(probability_condition, max_computation_condition)
            # ... for the entire batch.
            return tf.reduce_any(combined_conditions)

        # This actually does the computation of self.adaptive_memory_hop,
        # checking the condition at every step to see if it should stop.
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
        # We don't want to mask either of the outputs here, so we return None for both of them.
        return [None, None]

    def get_output_shape_for(self, input_shapes):
        # We output two tensors from this layer, the final memory representation and
        # the attended knowledge from the final memory network step. Both have the same
        # shape as the initial memory vector (samples, encoding_dim) which is passed in as the
        # 2nd argument, so we return this shape twice.
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

        # First things first: let's actually do a memory network step. This is exactly the same as in the
        # vanilla memory network. We have to re-assign this as a method for this class because we can't
        # pass in functions to this method (adaptive_memory_hop) as we use it in the tf.while_loop, which
        # requires tensor-only arguments.
        current_memory, attended_knowledge = self.memory_step(encoded_question, previous_memory, encoded_knowledge)

        # Here, we are computing the probability that each sample in the batch will halt at this iteration.
        # This outputs a vector of probabilities of shape (samples, ).
        with tf.variable_scope("halting_calculation"):
            halting_probability = tf.squeeze(tf.sigmoid(K.dot(current_memory, self.halting_weight)), 1)

        # This is where the loop condition variables are controlled, which takes several steps.
        # First, we compute a new batch mask, which will be of size (samples, ). We want there to be 0s where
        # a given samples adaptive loop should have halted. To check this, we compare element-wise the previous mask
        # plus this iteration's halting probabilities to see if they are less than 1 - epsilon. Additionally, if a
        # given sample had halted at the previous batch, we don't want these to accidentally start again in this
        # iteration, so we also compare to the previous batch_mask using logical and.

        # Example of why we need to protect against the above scenario:
        # If we were at 0.8 and generated a probability of 0.3, which would take us over 1 - epsilon. We then don't add
        # this to the halting_accumulator, and then in the next iteration, we generate 0.1, which would not take us over
        # the limit, as the halting_accumulator is still at 0.8. However, we don't want to consider this contribution,
        # as we have already halted.
        new_batch_mask = tf.logical_and(
            tf.less(halting_accumulator + halting_probability, self.one_minus_epsilon),
            batch_mask)

        # Next, we update the halting_accumulator by adding on the halting_probabilities from this iteration, masked
        # by the new_batch_mask. Note that this means that if the halting_probability for a given sample has caused
        # the accumulator to go over 1 - epsilon, we DO NOT update this value in the halting_accumulator. Values in
        # this accumulator can never be over 1 - epsilon.
        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        halting_accumulator += halting_probability * new_float_mask

        # Finally, we update the halting_accumulator_for_comparison, which is only used in the halting condition in
        # the while_loop. Note that here, we are adding on the halting probabilities multiplied by the previous
        # iteration's batch_mask, which means that we DO update samples over 1 - epsilon. This means that we can check
        # in the loop condition to see if all samples are over 1 - epsilon, which means we should halt the while_loop.
        halting_accumulator_for_comparison += halting_probability * tf.cast(batch_mask, tf.float32)

        def use_probability():
            masked_halting_probability = tf.expand_dims(halting_probability * new_float_mask, 1)
            accumulated_memory_update = (current_memory * masked_halting_probability) + memory_accumulator
            return accumulated_memory_update

        def use_remainder():
            # When this function is called,
            remainder = tf.expand_dims(1.0 - halting_accumulator, 1)
            accumulated_memory_update = (current_memory * remainder) + memory_accumulator
            return accumulated_memory_update

        # This just counts the number of memory network steps we take for each sample.
        # We use this for regularisation - by adding this to the loss function, we can bias
        # the network to take fewer steps.
        hop_counter += new_float_mask

        # This condition checks whether a sample has gone over the permitted number of memory steps.
        counter_condition = tf.less(hop_counter, self.max_computation)

        # If any of the samples are (under the max number of steps AND not yet halted), we take the first
        # option in the conditional below. This option is just accumulating the memory network state, as the
        # output of this whole loop is a weighted sum of the memory representations with respect to the
        # halting probabilities at each step.

        # The second function is only used on the step before the while loop halts. This means that all of the
        # batches have finished (new_batch_mask will be all zeros, or we have reached the maximum number of steps).
        # For the final step, in order to make the weighted sum we are accumulating be an expected value (where all
        # the values sum to 1), we need to multiply by 1 - halting_accumulator. The reason for this is due to the
        # 1 - epsilon halting condition, as the final probability also needs to take into account this epsilon value.
        final_iteration_condition = tf.reduce_any(tf.logical_and(new_batch_mask, counter_condition))

        memory_accumulator = tf.cond(final_iteration_condition, use_probability, use_remainder)

        # We have to return all of these values as a requirement of the tf.while_loop. Some of them, we haven't updated,
        # such as the encoded_question and encoded_knowledge.
        return [new_batch_mask,
                halting_accumulator,
                halting_accumulator_for_comparison,
                hop_counter,
                encoded_question,
                current_memory,
                encoded_knowledge,
                memory_accumulator,
                attended_knowledge]












