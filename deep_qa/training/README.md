# Trainers

`Trainers` specify data, a model, and a way to train the model with the data.  This module groups
all of the common code related to these things, making only minimal assumptions about what kind of
data you're using or what the structure of your model is.  Really, a `Trainer` is just a nicer
interface to a Keras `Model`, we just call it something else to not create too much naming
confusion, and because the `Trainer` class provides a lot of functionality around training the
model that a Keras `Model` doesn't.
