r"""
It turns out that Keras' design is somewhat crazy\*, and there is no list of
optimizers that you can just import from Keras. So, this module specifies a
list, and a helper function or two for dealing with optimizer parameters.
Unfortunately, this means that we have a list that must be kept in sync with
Keras. Oh well.

\* Have you seen their get_from_module() method? See here:
https://github.com/fchollet/keras/blob/6e42b0e4a77fb171295b541a6ae9a3a4a79f9c87/keras/utils/generic_utils.py#L10.
That method means I could pass in 'clip_norm' as an optimizer, and it would try
to use that function as an optimizer. It also means there is no simple list of
implemented optimizers I can grab.

\* I should also note that Keras is an incredibly useful library that does a lot
of things really well. It just has a few quirks...
"""
from typing import Union

from keras import backend as K

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from ..common.params import Params


if K.backend() == "tensorflow":
    from keras.optimizers import Optimizer
    from tensorflow.python.training import adagrad, adadelta, adam, gradient_descent, rmsprop, optimizer
    from tensorflow import clip_by_value, clip_by_norm


    class TFOptimizer(Optimizer):
        """
        Wrapper class for native TensorFlow optimizers. This exists already in Keras,
        but it doesn't support the same API as the other optimisers, which, you know,
        would be nice.
        """
        def __init__(self, optimizer, **kwargs):

            self.clip_norm = kwargs.pop("clipnorm", None)
            self.clip_value = kwargs.pop("clipvalue", None)

            if kwargs != {}:
                raise TypeError('Unexpected keyword argument/s '
                                'passed to optimizer: ' + str(kwargs))

            self.optimizer = optimizer
            self.iterations = K.variable(0., name='iterations')
            self.updates = []

        def get_updates(self, params, constraints, loss):
            if constraints:
                raise ValueError('TF optimizers do not support '
                                 'weights constraints. Either remove '
                                 'all weights constraints in your model, '
                                 'or use a Keras optimizer.')
            grads = self.optimizer.compute_gradients(loss, params)

            if self.clip_norm is not None:
                grads = [clip_by_norm(grad, self.clip_norm) for grad in grads]
            if self.clip_value is not None:
                grads = [clip_by_value(grad, -self.clip_value, self.clip_value) for grad in grads]
            opt_update = self.optimizer.apply_gradients(
                grads, global_step=self.iterations)

            self.updates.append(opt_update)
            return self.updates

        @property
        def weights(self):
            raise NotImplementedError

        def get_config(self):
            raise NotImplementedError

        def from_config(self, config):
            raise NotImplementedError

    def callable_wrapper(optimiser: optimizer.Optimizer, name: str):
        """
        This allows arguments to be unpacked into the tensorflow
        optimiser if required, but first splits off the "clipnorm" and "clipvalue"
        optional arguments and passes those to the ``TFOptimizer`` wrapper class.
        This makes the interface to Tensorflow optimisers identical to Keras ones.

        :param optimiser: An instance of a tensorflow optimiser.
        :param name: The name of the optimiser class. Just used to get default learning rates
          if they aren't present.
        :return: A callable optimiser which takes a dictionary of inputs.
        """
        # Keras has default values for all of the optimiser classes, but tensorflow is
        # missing some. Here we set them to the default values(from Keras) if they
        # are not passed as arguments.
        default_learning_rates = {"sgd": 0.01, "rmsprop": 0.001, "adagrad": 0.01}

        def callable_optimiser(**kwargs):
            clipnorm = kwargs.pop("clipnorm", None)
            clipvalue = kwargs.pop("clipvalue", None)
            learning_rate = kwargs.get("learning_rate", None)

            if learning_rate is None and name in default_learning_rates.keys():
                kwargs["learning_rate"] = default_learning_rates[name]

            return TFOptimizer(optimiser(**kwargs), clipnorm=clipnorm, clipvalue=clipvalue)
        return callable_optimiser

    optimizers = {  # pylint: disable=invalid-name
        'sgd': callable_wrapper(gradient_descent.GradientDescentOptimizer, "sgd"),
        'rmsprop': callable_wrapper(rmsprop.RMSPropOptimizer, "rmsprop"),
        'adagrad': callable_wrapper(adagrad.AdagradOptimizer, "adagrad"),
        'adadelta': callable_wrapper(adadelta.AdadeltaOptimizer, "adadelta"),
        'adam': callable_wrapper(adam.AdamOptimizer, "adam"),
        'adamax': Adamax,
        'nadam': Nadam,
    }
else:

    optimizers = {  # pylint: disable=invalid-name
            'sgd': SGD,
            'rmsprop': RMSprop,
            'adagrad': Adagrad,
            'adadelta': Adadelta,
            'adam': Adam,
            'adamax': Adamax,
            'nadam': Nadam,
            }


def optimizer_from_params(params: Union[Params, str]):
    """
    This method converts from a parameter object like we use in our Trainer
    code into an optimizer object suitable for use with Keras. The simplest
    case for both of these is a string that shows up in `optimizers` above - if
    `params` is just one of those strings, we return it, and everyone is happy.
    If not, we assume `params` is a Dict[str, Any], with a "type" key, where
    the value for "type" must be one of those strings above. We take the rest
    of the parameters and pass them to the optimizer's constructor.

    """
    if isinstance(params, str) and params in optimizers.keys():
        return params
    optimizer = params.pop_choice("type", optimizers.keys())
    return optimizers[optimizer](**params)
