About Trainers
==============

.. include:: ../../TRAINING.rst

On top of ``Trainer``, which is a nicer interface to a Keras ``Model``, this module provides a
``TextTrainer``, which adds a lot of functionality for building Keras ``Models`` that work with
text.  We provide APIs around word embeddings, sentence encoding, reading and padding datasets, and
similar things.  All of the concrete models that we have so far in DeepQA inherit from
``TextTrainer``, so understanding how to use this class is pretty important to understanding
DeepQA.

We also deal with the notion of pre-training in this module. A Pretrainer is a Trainer that depends
on another Trainer, building its model using pieces of the enclosed Trainer, so that training the
Pretrainer updates the weights in the enclosed Trainer object.
