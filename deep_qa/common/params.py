from typing import Any, Dict, List

import logging
import pyhocon

from .checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PARAMETER = 60
logging.addLevelName(PARAMETER, "PARAM")


def param(self, message, *args, **kws):
    """
    Add a method to logger which allows us
    to always log parameters unless you set the logging
    level to be higher than 60 (which is higher than the
    standard highest level of 50, corresponding to WARNING).
    """
    # Logger takes its '*args' as 'args'.
    if self.isEnabledFor(PARAMETER):
        self._log(PARAMETER, message, args, **kws) # pylint: disable=protected-access
logging.Logger.param = param


def _get_choice_error_message(value: Any, choices: List[Any], name: str=None) -> str:
    if name:
        return '%s not in acceptable choices for %s: %s' % (value, name, str(choices))
    else:
        return '%s not in acceptable choices: %s' % (value, str(choices))


def pop_with_logging(params: Dict[str, Any], key: str):
    """
    Performs the functionality associated with dict.pop(key) but with parameter
    logging. This is required because pop_with_default may receive a default value
    of None, which means we can't check for it not being passed.
    """
    value = params.pop(key)
    logger.param(key + " : " + str(value))
    return value


def pop_with_default(params: Dict[str, Any], key: str, default: Any):

    """
    Performs the functionality associated with dict.pop(key, default) but with parameter
    logging.
    """
    value = params.pop(key, default)
    logger.param(key + " : " + str(value))
    return value


def get_choice(params: Dict[str, Any], key: str, choices: List[Any], name: str=None):
    """
    Gets the value of `key` in the `params` dictionary, ensuring that the value is one of the given
    choices.  `name` is an optional description for where a configuration error happened, if there
    is one.

    Note that this _pops_ the key from params, modifying the dictionary, consistent with how
    parameters are processed in this codebase.
    """

    value = pop_with_logging(params, key)
    if value not in choices:
        raise ConfigurationError(_get_choice_error_message(value, choices, name))
    return value


def get_choice_with_default(params: Dict[str, Any],
                            key: str,
                            choices: List[Any],
                            default: Any=None,
                            name: str=None):
    """
    Like get_choice, but with a default value.  If `default` is None, we use the first item in
    `choices` as the default.
    """
    try:
        return get_choice(params, key, choices, name)
    except KeyError:
        if default is None:
            return choices[0]
        if default not in choices:
            raise ConfigurationError(_get_choice_error_message(default, choices, name))
        return default


def assert_params_empty(params: Dict[str, Any], class_name: str):
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
