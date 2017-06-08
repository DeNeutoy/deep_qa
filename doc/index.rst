.. deep_qa documentation master file, created by
   sphinx-quickstart on Wed Jan 25 11:35:07 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

.. include:: ../README.rst


.. toctree::
   :hidden:

   self
   run
   releases

.. toctree::
   :caption: Training
   :hidden:

   training/about_trainers
   training/trainer
   training/text_trainer
   training/multi_gpu
   training/misc

.. toctree::
   :caption: Data
   :hidden:

   data/about_data
   data/instances
   data/entailment
   data/multiple_choice_qa
   data/reading_comprehension
   data/sentence_selection
   data/sequence_tagging
   data/text_classification
   data/wrappers
   data/tokenizers
   data/data_generator
   data/general_data_utils

.. toctree::
   :caption: Models
   :hidden:

   models/about_models
   models/entailment
   models/memory_networks
   models/multiple_choice_qa
   models/sentence_selection
   models/reading_comprehension
   models/text_classification

.. toctree::
   :caption: Layers
   :hidden:

   layers/about_layers
   layers/core_layers
   layers/attention
   layers/backend
   layers/encoders
   layers/entailment_models
   layers/tuple_matchers
   layers/wrappers

.. toctree::
   :caption: Tensor Utils
   :hidden:

   tensors/about_tensors
   tensors/core_tensors
   tensors/similarity_functions

.. toctree::
   :caption: Common Utils
   :hidden:

   common/about_common
   common/checks
   common/params
