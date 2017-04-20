from typing import Any, Dict, List, Union
from collections import MutableMapping

import logging
import pyhocon

from overrides import overrides
from .checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PARAMETER = 60
logging.addLevelName(PARAMETER, "PARAM")


def __param(self, message, *args, **kws):
    """
    Add a method to logger which allows us
    to always log parameters unless you set the logging
    level to be higher than 60 (which is higher than the
    standard highest level of 50, corresponding to WARNING).
    """
    # Logger takes its '*args' as 'args'.
    if self.isEnabledFor(PARAMETER):
        self._log(PARAMETER, message, args, **kws) # pylint: disable=protected-access
logging.Logger.param = __param


class Params(MutableMapping):
    """
    A Class representing a parameter dictionary with a history. Using this allows
    exact reproduction of a parameter file used in an experiment from logs, even when
    default values are used.
    """

    # This allows us to check for the presence of "None" as a default argument,
    # which we require because we make a distinction bewteen passing a value of "None"
    # and passing no value to the default parameter of "pop".
    DEFAULT = object()

    def __init__(self, params: Dict[str, Any], history: str=""):

        self.params = params
        self.history = history

    @overrides
    def pop(self, key: str, default: Any=DEFAULT) -> Union["Params", Any]:
        """
        Performs the functionality associated with dict.pop(key) but with parameter
        logging. This is required because pop_with_default may receive a default value
        of None, which means we can't check for it not being passed.
        """
        if default is self.DEFAULT:
            value = self.params.pop(key)
        else:
            value = self.params.pop(key, default)
        logger.param(key + " : " + str(value))
        return self.__check_is_dict(key, value)

    @overrides
    def get(self, key: str, default: Any=DEFAULT) -> Union["Params", Any]:
        """
        Performs the functionality associated with dict.pop(key) but with parameter
        logging. This is required because pop_with_default may receive a default value
        of None, which means we can't check for it not being passed.
        """
        print(default)
        if default is self.DEFAULT:
            value = self.params.pop(key)
        else:
            value = self.params.pop(key, default)
        return self.__check_is_dict(key, value)

    def pop_choice(self, key: str, choices: List[Any], name: str=None) -> Union["Params", Any]:
        """
        Gets the value of `key` in the `params` dictionary, ensuring that the value is one of the given
        choices.  `name` is an optional description for where a configuration error happened, if there
        is one.

        Note that this _pops_ the key from params, modifying the dictionary, consistent with how
        parameters are processed in this codebase.
        """

        value = self.pop(key)
        if value not in choices:
            raise ConfigurationError(self._get_choice_error_message(value, choices, name))
        return self.__check_is_dict(key, value)

    def pop_choice_with_default(self,
                                key: str,
                                choices: List[Any],
                                default: Any=None,
                                name: str=None) -> Union["Params", Any]:
        """
        Like get_choice, but with a default value.  If `default` is None, we use the first item in
        `choices` as the default.
        """
        try:
            return self.pop_choice(key, choices, name)
        except KeyError:
            if default is None:
                default = choices[0]
            if default not in choices:
                raise ConfigurationError(self._get_choice_error_message(default, choices, name))

            return self.__check_is_dict(key, default)

    def __check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = new_history if self.history == "" else self.history + "." + new_history
            return Params(value, new_history)
        else:
            return value

    @staticmethod
    def _get_choice_error_message(value: Any, choices: List[Any], name: str=None) -> str:
        if name:
            return '%s not in acceptable choices for %s: %s' % (value, name, str(choices))
        else:
            return '%s not in acceptable choices: %s' % (value, str(choices))

    def __getitem__(self, key):
        return self.__check_is_dict(key, self.params[key])

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)


def assert_params_empty(params: "Params", class_name: str):
    """
    Raises a ConfigurationError if ``params`` is not empty, with a message about where the extra
    parameters were passed to.
    """
    if len(params) != 0:
        raise ConfigurationError("Extra parameters passed to {}: {}".format(class_name, params))


def replace_none(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    for key in dictionary.keys():
        if dictionary[key] == "None":
            dictionary[key] = None
        elif isinstance(dictionary[key], pyhocon.config_tree.ConfigTree):
            dictionary[key] = replace_none(dictionary[key])
    return dictionary
