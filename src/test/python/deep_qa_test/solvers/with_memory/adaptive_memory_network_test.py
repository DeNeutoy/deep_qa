
# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.solvers.with_memory.memory_network import MemoryNetworkSolver
from ...common.constants import TEST_DIR
from ...common.solvers import get_solver
from ...common.solvers import write_memory_network_files
from ...common.test_markers import requires_tensorflow


@requires_tensorflow
class TestMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        args = {'recurrence_mode': {'type': 'adaptive'}}
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()