from typing import Any, Dict
from collections import OrderedDict

from keras import backend as K


class FixedRecurrence:
    '''
    This recurrence class simply performs a fixed number of memory network steps and
    returns the memory representation and representation of the background knowledge
    generated by the knowledge_selector and knowledge_combiner layers (the simplest
    case being a weighted sum).
    '''
    def __init__(self, memory_network, params: Dict[str, Any]):

        self.num_memory_layers = params.pop("num_memory_layers", 1)
        self.memory_network = memory_network

    def __call__(self, encoded_question, current_memory, encoded_background):
        for _ in range(self.num_memory_layers):
            current_memory, attended_knowledge = \
                self.memory_network.memory_step(encoded_question, current_memory, encoded_background)
        return current_memory, attended_knowledge

recurrence_modes = OrderedDict()  # pylint: disable=invalid-name
recurrence_modes["fixed"] = FixedRecurrence
if K.backend() == 'tensorflow':
    from .adaptive_recurrence import AdaptiveRecurrence
    recurrence_modes["adaptive"] = AdaptiveRecurrence
