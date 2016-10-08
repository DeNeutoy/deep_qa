from copy import deepcopy
from typing import Dict, Any
from overrides import overrides

from keras import backend as K
from keras.layers import TimeDistributed, Dropout, merge
from keras.models import Model

from ..common.params import get_choice_with_default
from ..data.dataset import Dataset, IndexedDataset, TextDataset  # pylint: disable=unused-import
from ..data.text_instance import TrueFalseInstance
from ..layers.knowledge_selectors import selectors
from ..layers.memory_updaters import updaters
from ..layers.entailment_models import entailment_models, entailment_input_combiners
from ..layers.knowledge_combiners import knowledge_combiners
from .nn_solver import NNSolver
from .pretraining.snli_pretrainer import SnliAttentionPretrainer, SnliEntailmentPretrainer


# TODO(matt): make this class abstract, and make a TrueFalseMemoryNetwork subclass.
class MemoryNetworkSolver(NNSolver):
    '''
    We call this a Memory Network Solver because it has an attention over background knowledge, or
    "memory", similar to a memory network.  This implementation generalizes the architecture of the
    original memory network, though, and can be used to implement several papers in the literature,
    as well as some models that we came up with.

    Our basic architecture is as follows:
        Input: a sentence encoding and a set of background knowledge ("memory") encodings

        current_memory = sentence_encoding
        For each memory layer:
           attention_weights = knowledge_selector(current_memory, background)
           aggregated_background = weighted_sum(attention_weights, background)
           current_memory = memory_updater(current_memory, aggregated_background)
        final_score = entailment_model(aggregated_background, current_memory, sentence_encoding)

    There are thus three main knobs that can be turned (in addition to the number of memory
    layers):
        1. the knowledge_selector
        2. the memory_updater
        3. the entailment_model

    The original memory networks paper used the following:
        1. dot product (our DotProductKnowledgeSelector)
        2. sum
        3. linear classifier on top of current_memory

    The attentive reader in "Teaching Machines to Read and Comprehend", Hermann et al., 2015, used
    the following:
        1. a dense layer with a dot product bias (our ParameterizedKnowledgeSelector)
        2. Dense(K.concat([current_memory, aggregated_background]))
        3. Dense(current_memory)

    Our thought is that we should treat the last step as an entailment problem - does the
    background knowledge entail the input sentence?  Previous work was solving a different problem,
    so they used simpler models "entailment".
    '''

    def __init__(self, params: Dict[str, Any]):
        self.train_background = params.pop('train_background', None)
        self.positive_train_background = params.pop('positive_train_background', None)
        self.negative_train_background = params.pop('negative_train_background', None)
        self.validation_background = params.pop('validation_background', None)
        self.test_background = params.pop('test_background', None)
        self.debug_background = params.pop('debug_background', None)

        self.num_memory_layers = params.pop('num_memory_layers', 1)

        # We need to pop these parameters now, but use them after we've called the superclass
        # constructor.  We don't need to save them to self; see below.
        pretrain_entailment = params.pop('pretrain_entailment', False)
        pretrain_attention = params.pop('pretrain_attention', False)
        snli_file = params.pop('snli_file', None)

        # These parameters specify the kind of knowledge selector, used to compute an attention
        # over a collection of background information.
        # If given, this must be a dict.  We will use the "type" key in this dict (which must match
        # one of the keys in `selectors`) to determine the type of the selector, then pass the
        # remaining args to the selector constructor.
        self.knowledge_selector_params = params.pop('knowledge_selector', {})
        self.knowledge_combiner_params = params.pop('knowledge_combiner', {})

        # Same deal with these three as with the knowledge selector params.
        self.memory_updater_params = params.pop('memory_updater', {})
        self.entailment_combiner_params = params.pop('entailment_input_combiner', {})
        self.entailment_model_params = params.pop('entailment_model', {})

        # Now that we've processed all of our parameters, we can call the superclass constructor.
        # The superclass will check that there are no unused parameters, so we need to call this
        # _after_ we've popped everything we use.
        super(MemoryNetworkSolver, self).__init__(params)

        # self.pretrainers gets set in the superclass constructor, so now we can append to it.
        if pretrain_entailment:
            self.pretrainers.append(SnliEntailmentPretrainer(self, snli_file))
        if pretrain_attention:
            self.pretrainers.append(SnliAttentionPretrainer(self, snli_file))

        # These are the entailment models that are compatible with this solver.
        self.entailment_choices = ['true_false_mlp']

        # This specifies whether the entailment decision made my this solver (if any) has a sigmoid
        # activation or a softmax activation.  This value is read by some pre-trainers, which need
        # to know how to construct data for training a model.
        self.has_sigmoid_entailment = False

        # Model-specific variables that will get set and used later.  For many of these, we don't
        # want to set them now, because they use max length information that only gets set after
        # reading the training data.
        self.knowledge_selector_layers = {}
        self.memory_updater_layers = {}
        self.entailment_input_combiner = None
        self.entailment_model = None
        self.max_knowledge_length = None

    @overrides
    def can_train(self) -> bool:
        has_train_background = (self.train_background is not None) or (
                self.positive_train_background is not None and
                self.negative_train_background is not None)
        has_validation_background = self.validation_background is not None
        has_background = has_train_background and has_validation_background
        return has_background and super(MemoryNetworkSolver, self).can_train()

    @overrides
    def can_test(self) -> bool:
        return self.test_background is not None and super(MemoryNetworkSolver, self).can_test()

    @overrides
    def _instance_type(self):
        return TrueFalseInstance

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(MemoryNetworkSolver, cls)._get_custom_objects()
        for object_dict in [updaters, selectors, entailment_input_combiners]:
            for value in object_dict.values():
                custom_objects[value.__name__] = value
        return custom_objects

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {
                'word_sequence_length': self.max_sentence_length,
                'background_sentences': self.max_knowledge_length,
                }

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_knowledge_length = max_lengths['background_sentences']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]
        self.max_knowledge_length = self.model.get_input_shape_at(0)[1][1]

    def _get_question_shape(self):
        """
        This is the shape of the input word sequences for a question, not including the batch size.
        """
        return (self.max_sentence_length,)

    def _get_background_shape(self):
        """
        This is the shape of background data (word sequences) associated with a question, not
        including the batch size.
        """
        return (self.max_knowledge_length, self.max_sentence_length)

    def _get_knowledge_axis(self):
        """
        We need to merge and concatenate things in the course of the memory network, and we do it
        in the knowledge_length dimension.  This tells us which axis that dimension is in,
        including the batch_size.

        So, for the true/false memory network, which has background input shape
        (batch_size, knowledge_length, sentence_length), this would be 1.  For the multiple choice
        memory network, which has background input shape
        (batch_size, num_options, knowledge_length, sentence_length), this would be 2.
        """
        # pylint: disable=no-self-use
        return 1

    def _get_merged_background_shape(self):
        """
        This method returns a lambda function, which takes input the shape of the question encoding
        and the knowledge encoding, and returns as output the shape of the merged question and
        background encodings.  This merge just stacks the question encoding on top of the
        background encoding, adding one to the knowledge_length dimension.
        """
        knowledge_axis = self._get_knowledge_axis()
        def merged_shape(input_shapes):
            background_shape = [x for x in input_shapes[1]]
            background_shape[knowledge_axis] += 1
            return tuple(background_shape)
        return merged_shape

    def _get_knowledge_selector(self, layer_num: int):
        """
        Instantiates a KnowledgeSelector layer.  This is an overridable method because some
        subclasses might need to TimeDistribute this, for instance.
        """
        if layer_num not in self.knowledge_selector_layers:
            layer = self._get_new_knowledge_selector(name='knowledge_selector_%d' % layer_num)
            self.knowledge_selector_layers[layer_num] = layer
        return self.knowledge_selector_layers[layer_num]

    def _get_new_knowledge_selector(self, name='knowledge_selector'):
        # The code that follows would be destructive to self.knowledge_selector_params (lots of
        # calls to params.pop()), but it's possible we'll want to call this more than once.  So
        # we'll make a copy and use that instead of self.knowledge_selector_params.
        params = deepcopy(self.knowledge_selector_params)
        selector_type = get_choice_with_default(params, "type", list(selectors.keys()))
        params['name'] = name
        return selectors[selector_type](**params)

    def _get_knowledge_combiner(self, name='knowledge_combiner'):
        # The code that follows would be destructive to self.knowledge_combiner_params (lots of
        # calls to params.pop()), but it's possible we'll want to call this more than once.  So
        # we'll make a copy and use that instead of self.knowledge_combiner_params.
        # TODO: this should only be called once. How to check this?
        params = deepcopy(self.knowledge_combiner_params)
        params["knowledge_axis"] = self._get_knowledge_axis()
        combiner_type = get_choice_with_default(params, "type", list(knowledge_combiners.keys()))

        return knowledge_combiners[combiner_type](**params)

    def _get_memory_updater(self, layer_num: int):
        """
        Instantiates a MemoryUpdater layer.  This is an overridable method because some subclasses
        might need to TimeDistribute this, for instance.
        """
        if layer_num not in self.memory_updater_layers:
            layer = self._get_new_memory_updater(name='memory_updater_%d' % layer_num)
            self.memory_updater_layers[layer_num] = layer
        return self.memory_updater_layers[layer_num]

    def _get_new_memory_updater(self, name='memory_updater'):
        # The code that follows would be destructive to self.memory_updater_params (lots of calls
        # to params.pop()), but it's possible we'll want to call this more than once.  So we'll
        # make a copy and use that instead of self.memory_updater_params.
        params = deepcopy(self.memory_updater_params)
        updater_type = get_choice_with_default(params, "type", list(updaters.keys()))
        params['name'] = name
        params['encoding_dim'] = self.embedding_size
        return updaters[updater_type](**params)

    def _get_entailment_input_combiner(self):
        """
        Instantiates an EntailmentCombiner layer.  This is an overridable method because some
        subclasses might need to TimeDistribute this, for instance.
        """
        if self.entailment_input_combiner is None:
            self.entailment_input_combiner = self._get_new_entailment_input_combiner()
        return self.entailment_input_combiner

    def _get_new_entailment_input_combiner(self):
        # The code that follows would be destructive to self.entailment_combiner_params (lots of
        # calls to params.pop()), but it's possible we'll want to call this more than once.  So
        # we'll make a copy and use that instead of self.entailment_combiner_params.
        params = deepcopy(self.entailment_combiner_params)
        params['encoding_dim'] = self.embedding_size
        combiner_type = get_choice_with_default(params, "type", list(entailment_input_combiners.keys()))
        return entailment_input_combiners[combiner_type](**params)

    def _get_entailment_output(self, combined_input):
        """
        Gets from the combined entailment input to an output that matches the training labels.
        This is typically done using self.entailment_model.classify(), but could do other things
        also.

        To allow for subclasses to take additional inputs in the entailment model, the return value
        is a tuple of ([additional input layers], output layer).  For instance, this is where
        answer options go, for models that separate the question text from the answer options.
        """
        return [], self._get_entailment_model().classify(combined_input)

    def _get_entailment_model(self):
        if self.entailment_model is None:
            self.entailment_model = self._get_new_entailment_model()
        return self.entailment_model

    def _get_new_entailment_model(self):
        # The code that follows would be destructive to self.entailment_model_params (lots of calls
        # to params.pop()), but it's possible we'll want to call this more than once.  So we'll
        # make a copy and use that instead of self.entailment_model_params.
        entailment_params = deepcopy(self.entailment_model_params)
        model_type = get_choice_with_default(entailment_params, "type", self.entailment_choices)
        # TODO(matt): Not great to have these two lines here.
        if model_type == 'question_answer_mlp':
            entailment_params['answer_dim'] = self.embedding_size
        return entailment_models[model_type](entailment_params)

    @overrides
    def _build_model(self):
        # Steps 1 and 2: Convert inputs to sequences of word vectors, for both the question
        # inputs and the knowledge inputs.
        question_input_layer, question_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_question_shape(), name_prefix="sentence")
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_background_shape(), name_prefix="background")

        # Step 3: Encode the two embedded inputs using the sentence encoder.
        question_encoder = self._get_sentence_encoder()

        # Knowledge encoder will have the same encoder running on a higher order tensor.
        # i.e., question_encoder: (samples, num_words, word_dim) -> (samples, word_dim)
        # and knowledge_encoder: (samples, knowledge_len, num_words, word_dim) ->
        #                       (samples, knowledge_len, word_dim)
        # TimeDistributed generally loops over the second dimension.
        knowledge_encoder = TimeDistributed(question_encoder, name='knowledge_encoder')
        encoded_question = question_encoder(question_embedding)  # (samples, word_dim)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)  # (samples, knowledge_len, word_dim)

        # Step 4: Merge the two encoded representations and pass into the knowledge backed scorer.
        # At each step in the following loop, we take the question encoding, or the output of
        # the previous memory layer, merge it with the knowledge encoding and pass it to the
        # current memory layer.
        current_memory = encoded_question

        knowledge_combiner = self._get_knowledge_combiner()
        knowledge_axis = self._get_knowledge_axis()
        for i in range(self.num_memory_layers):
            # We want to merge a matrix and a tensor such that the new tensor will have one
            # additional row (at the beginning) in all slices.
            # (samples, word_dim) + (samples, knowledge_len, word_dim)
            #       -> (samples, 1 + knowledge_len, word_dim)
            # Since this is an unconventional merge, we define a customized lambda merge.
            # Keras cannot infer the shape of the output of a lambda function, so we make
            # that explicit.
            merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=knowledge_axis),
                                                           layer_outs[1]],
                                                          axis=knowledge_axis)
            merged_shape = self._get_merged_background_shape()
            merged_encoded_rep = merge([current_memory, encoded_knowledge],
                                       mode=merge_mode,
                                       output_shape=merged_shape,
                                       name='concat_question_with_background_%d' % i)

            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_selector = self._get_knowledge_selector(i)
            attention_weights = knowledge_selector(regularized_merged_rep)
            # Defining weighted average as a custom merge mode. Takes two inputs: data and weights
            # ndim of weights is one less than data.

            # We now combine the knowledge and the weights using the knowledge combiner.
            attended_knowledge = knowledge_combiner(encoded_knowledge, attention_weights, i)

            # To make this easier to TimeDistribute, we're going to concatenate the current memory
            # with the attended knowledge, and use that as the input to the memory updater, instead
            # of just passing a list.
            # We going from two inputs of (batch_size, encoding_dim) to one input of (batch_size,
            # encoding_dim * 2).
            updater_input = merge([current_memory, attended_knowledge],
                                  mode='concat',
                                  concat_axis=knowledge_axis,
                                  name='concat_current_memory_with_background_%d' % i)
            memory_updater = self._get_memory_updater(i)
            current_memory = memory_updater(updater_input)


        # TODO(matt): we now have subclasses that do answer selection, instead of entailment, and
        # it's not very nice to shoehorn them into the same "entailment" model.  It would be nice
        # to generalize this into some "final output" section, but I'm not sure how to do that
        # cleanly.

        # Step 5: Finally, run the sentence encoding, the current memory, and the attended
        # background knowledge through an entailment model to get a final true/false score.
        entailment_input = merge([encoded_question, current_memory, attended_knowledge],
                                 mode='concat',
                                 concat_axis=knowledge_axis,
                                 name='concat_entailment_inputs')
        combined_input = self._get_entailment_input_combiner()(entailment_input)
        extra_entailment_inputs, entailment_output = self._get_entailment_output(combined_input)

        # Step 6: Define the model, and return it. The model will be compiled and trained by the
        # calling method.
        input_layers = [question_input_layer, knowledge_input_layer]
        input_layers.extend(extra_entailment_inputs)
        memory_network = Model(input=input_layers, output=entailment_output)
        return memory_network

    @overrides
    def _get_training_data(self):
        instance_type = self._instance_type()
        if self.train_file:
            dataset = TextDataset.read_from_file(self.train_file, instance_type, tokenizer=self.tokenizer)
            background_dataset = TextDataset.read_background_from_file(dataset, self.train_background)
        else:
            positive_dataset = TextDataset.read_from_file(self.positive_train_file,
                                                          instance_type,
                                                          label=True,
                                                          tokenizer=self.tokenizer)
            positive_background = TextDataset.read_background_from_file(positive_dataset,
                                                                        self.positive_train_background)
            negative_dataset = TextDataset.read_from_file(self.negative_train_file,
                                                          instance_type,
                                                          label=False,
                                                          tokenizer=self.tokenizer)
            negative_background = TextDataset.read_background_from_file(negative_dataset,
                                                                        self.negative_train_background)
            background_dataset = positive_background.merge(negative_background)
        if self.max_training_instances is not None:
            background_dataset = background_dataset.truncate(self.max_training_instances)
        self.data_indexer.fit_word_dictionary(background_dataset)
        self.training_dataset = background_dataset
        return self.prep_labeled_data(background_dataset, for_train=True, shuffle=True)

    @overrides
    def _get_validation_data(self):
        dataset = TextDataset.read_from_file(self.validation_file, self._instance_type(), tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset, self.validation_background)
        self.validation_dataset = background_dataset
        return self._prep_question_dataset(background_dataset)

    @overrides
    def _get_test_data(self):
        dataset = TextDataset.read_from_file(self.test_file, self._instance_type(), tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset, self.test_background)
        return self._prep_question_dataset(background_dataset)

    @overrides
    def _get_debug_dataset_and_input(self):
        dataset = TextDataset.read_from_file(self.debug_file, self._instance_type(), tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset, self.debug_background)
        # Now get inputs, and ignore the labels (background_dataset has them)
        inputs, _ = self.prep_labeled_data(background_dataset, for_train=False, shuffle=False)
        return background_dataset, inputs

    def get_debug_layer_names(self):
        debug_layer_names = []
        for layer in self.model.layers:
            if "knowledge_selector" in layer.name:
                debug_layer_names.append(layer.name)
        return debug_layer_names

    def debug(self, debug_dataset, debug_inputs, epoch: int):
        """
        A debug_model must be defined by now. Run it on debug data and print the
        appropriate information to the debug output.
        """
        debug_output_file = open("%s_debug_%d.txt" % (self.model_prefix, epoch), "w")
        scores = self.score(debug_inputs)
        attention_outputs = self.debug_model.predict(debug_inputs)
        if self.num_memory_layers == 1:
            attention_outputs = [attention_outputs]
        # Collect values from all hops of attention for a given instance into attention_values.
        for instance, score, *attention_values in zip(debug_dataset.instances,
                                                      scores, *attention_outputs):
            sentence = instance.text
            background_info = instance.background
            label = instance.label
            positive_score = score[1]  # Only get p(t|x)
            # Remove the attention values for padding
            attention_values = [values[-len(background_info):] for values in attention_values]
            print("Sentence: %s" % sentence, file=debug_output_file)
            print("Label: %s" % label, file=debug_output_file)
            print("Assigned score: %.4f" % positive_score, file=debug_output_file)
            print("Weights on background:", file=debug_output_file)
            for i, background_i in enumerate(background_info):
                if i >= len(attention_values[0]):
                    # This happens when IndexedBackgroundInstance.pad() ignored some
                    # sentences (at the end). Let's ignore them too.
                    break
                all_hops_attention_i = ["%.4f" % values[i] for values in attention_values]
                print("\t%s\t%s" % (" ".join(all_hops_attention_i), background_i),
                      file=debug_output_file)
            print(file=debug_output_file)
        debug_output_file.close()
