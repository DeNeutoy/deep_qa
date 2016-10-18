

from copy import deepcopy
from typing import Any, Dict, List
from overrides import overrides


from keras import backend as K
from keras.layers import Dropout, merge
from keras.models import Model

from .memory_network import MemoryNetworkSolver
from ...common.params import get_choice_with_default
from ...data.dataset import Dataset, TextDataset
from ...data.instances.true_false_instance import TrueFalseInstance
from ...layers.knowledge_selectors import selectors
from ...layers.memory_updaters import updaters
from ...layers.entailment_models import entailment_models, entailment_input_combiners
from ...layers.knowledge_combiners import knowledge_combiners
from ...layers.knowledge_encoders import knowledge_encoders



class AdaptiveMemoryNetworkSolver(MemoryNetworkSolver):

    def __init__(self, params: Dict[str, Any]):

        self.epsilon = params.pop("epsilon", 0.01)
        self.max_computation = params.pop("max_computation", 10)

        super(AdaptiveMemoryNetworkSolver, self).__init__(params)

    def memory_hop(self):