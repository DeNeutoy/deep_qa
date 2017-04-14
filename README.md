[![Build
Status](https://api.travis-ci.org/allenai/deep_qa.svg?branch=master)](https://travis-ci.org/allenai/deep_qa)
[![Documentation
Status](https://readthedocs.org/projects/deep-qa/badge/?version=latest)](http://deep-qa.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/allenai/deep_qa/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/deep_qa)

# Deep QA

This repository contains code for training deep learning systems to do
question answering tasks. Our primary focus is on Aristo's science
questions, though we can run various models on several popular datasets.

This library is built on top of Keras (for actually training and
executing deep models) and is also designed to be used with an
experiment framework written in Scala, which you can find here: [Deep QA
Experiments](https://github.com/allenai/deep_qa_experiments).

## Running experiments with python

Although we recommend using the Deep QA Experiments library to run
reproducible experiments, this library is designed as standalone
software which runs deep NLP models using json specification files. To
do this, from the base directory, you run the command
`python src/main/python/run_solver.py [model_config]`. You must use
python &gt;= 3.5, as we make heavy use of the type annotations
introduced in python 3.5 to aid in code readability (I recommend using
[anaconda](https://www.continuum.io/downloads) to set up python 3, if
you don't have it set up already).

You can see some examples of what model configuration files look like in
the [example experiments
directory](https://github.com/allenai/deep_qa/tree/master/example_experiments).
We try to keep these up to date, but the way parameters are specified is
still sometimes in a state of flux, so we make no promises that these
are actually usable with the current master. Looking at the most
recently added or changed example experiment should be your best bet to
get an accurate format. If you find one that's out of date, submitting a
pull request to fix it would be great.

Finally, the way parameters are parsed in DeepQA can be a little
confusing. When you provide a json specification, various classes will
pop things from this dictionary of values (actually pop them, so they
aren't in the parameter dict any more). This is helpful because it
allows you to check that all of the parameters you pass are used at some
point, preventing hard to find bugs, as well as enabling clear
separation of functionality because there are no globally defined
variables, such as is often the case with other argument parsing
methods.

## Organisation

The deep\_qa library is organised into the following main sections:

-   Common: Code for parameter parsing, logging and runtime checks.
-   Contrib: Related code for experiments and untested layers, models
    and features. Generally untested.
-   Data: Indexing, padding, tokenisation, stemming, embedding and
    general dataset manipulation happens here.
-   Layers: The bulk of the library. Use these Layers to compose new
    models. Some of these Layers are very similar to what you might find
    in Keras, but altered slightly to support arbitrary dimensions or
    correct masking.
-   Models: Frameworks for different types of task. These generally all
    extend the TextTrainer class which provides training capabilities to
    a DeepQaModel. We have models for Sequence Tagging, Entailment,
    Multiple Choice QA, Reading Comprehension and more. Take a look at
    the README for more details.
-   Tensors: Convenience functions for writing the internals of Layers.
    Will almost exclusively be used inside Layer implementations.
-   Training: This module does the heavy lifting for training and
    optimisation. We also wrap the Keras Model class to give it some
    useful debugging functionality.

We've tried to also give reasonable documentation throughout the code,
both in docstring comments and in READMEs distributed throughout the
code packages, so browsing github should be pretty informative if you're
confused about something. If you're still confused about how something
works, open an issue asking to improve documentation of a particular
piece of the code (or, if you've figured it out after searching a bit,
submit a pull request containing documentation improvements that would
have helped you).

## Implemented models

This repository implements several variants of memory networks,
including the models found in these papers:

-   The original MemNN, from [Memory
    Networks](https://arxiv.org/abs/1410.3916), by Weston, Chopra and
    Bordes
-   [End-to-end memory
    networks](https://www.semanticscholar.org/paper/End-To-End-Memory-Networks-Sukhbaatar-Szlam/10ebd5c40277ecba4ed45d3dc12f9f1226720523),
    by Sukhbaatar and others (close, but still in progress)
-   [Dynamic memory
    networks](https://www.semanticscholar.org/paper/Ask-Me-Anything-Dynamic-Memory-Networks-for-Kumar-Irsoy/04ee77ef1143af8b19f71c63b8c5b077c5387855),
    by Kumar and others
-   DMN+, from [Dynamic Memory Networks for Visual and Textual Question
    Answering](https://www.semanticscholar.org/paper/Dynamic-Memory-Networks-for-Visual-and-Textual-Xiong-Merity/b2624c3cb508bf053e620a090332abce904099a1),
    by Xiong, Merity and Socher
-   The attentive reader, from [Teaching Machines to Read and
    Comprehend](https://www.semanticscholar.org/paper/Teaching-Machines-to-Read-and-Comprehend-Hermann-Kocisk%C3%BD/2cb8497f9214735ffd1bd57db645794459b8ff41),
    by Hermann and others
-   Gated Attention Reader from [Gated Attention Readers for Text
    Comprehension](https://www.semanticscholar.org/paper/Gated-Attention-Readers-for-Text-Comprehension-Dhingra-Liu/200594f44c5618fa4121be7197c115f78e6e110f),
-   Bidirectional Attention Flow, from [Bidirectional Attention Flow for
    Machine
    Comprehension](https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02),
-   Decomposable Attention, from [A Decomposable Attention Model for
    Natural Language
    Inference](https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27),
-   Windowed-memory MemNNs, from [The Goldilocks Principle: Reading
    Children's Books with Explicit Memory
    Representations](https://www.semanticscholar.org/paper/The-Goldilocks-Principle-Reading-Children-s-Books-Hill-Bordes/1ee46c3b71ebe336d0b278de9093cfca7af7390b)
    (in progress)

As well as some of our own, as-yet-unpublished variants. There is a lot
of similarity between the models in these papers, and our code is
structured in a way to allow for easily switching between these models.
As an example of this modular approach, here is a description of how
we've built an extensible memory network architecture in this library:
[this
readme.](./src/main/python/deep_qa/models/memory_networks/README.md) \#
Datasets

This code allows for easy experimentation with the following datasets:

-   [AI2 Elementary school science questions (no
    diagrams)](http://allenai.org/data.html)
-   [The Facebook Children's Book Test
    dataset](https://research.facebook.com/research/babi/)
-   [The Facebook bAbI
    dataset](https://research.facebook.com/research/babi/)
-   [The NewsQA dataset](https://datasets.maluuba.com/NewsQA)
-   [The Stanford Question Answering Dataset
    (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)
-   [The Who Did What dataset](https://tticnlp.github.io/who_did_what/)

And more to come... In the near future, we hope to also include easy
experimentation with [CNN/Daily Mail](http://cs.nyu.edu/~kcho/DMQA/) and
[SimpleQuestions](https://research.facebook.com/research/babi/).

## Contributing

If you use this code and think something could be improved, pull
requests are very welcome. Opening an issue is ok, too, but we're a lot
more likely to respond to a PR. The primary maintainer of this code is
[Matt Gardner](https://matt-gardner.github.io/), with a lot of help from
[Pradeep Dasigi](http://www.cs.cmu.edu/~pdasigi/) (who was the initial
author of this codebase), [Mark Neumann](http://markneumann.xyz/) and
[Nelson Liu](http://nelsonliu.me/).

## License

This code is released under the terms of the [Apache 2
license](https://www.apache.org/licenses/LICENSE-2.0).
