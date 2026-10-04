"""Unit tests for classical and learning-augmented page replacement algorithms."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.algorithms import FIFOPolicy, LRUPolicy, OptimalPolicy
from src.models import LogisticRegressionClassifier, DecisionTreeClassifierCustom
from src.workload import WorkloadGenerator
from src.simulator import Simulator


class TestClassicalAlgorithms(unittest.TestCase):
    """Verifies reference textbook trace behavior."""

    def test_fifo_canonical_trace(self):
        # Canonical textbook trace: [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1] with capacity 3
        trace = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        fifo = FIFOPolicy(capacity=3)
        for t, page in enumerate(trace):
            fifo.access(page, t)
        # Textbook standard: 15 page faults
        self.assertEqual(fifo.faults, 15)
        self.assertEqual(fifo.hits, 5)

    def test_lru_canonical_trace(self):
        trace = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        lru = LRUPolicy(capacity=3)
        for t, page in enumerate(trace):
            lru.access(page, t)
        # Textbook standard: 12 page faults
        self.assertEqual(lru.faults, 12)
        self.assertEqual(lru.hits, 8)

    def test_optimal_canonical_trace(self):
        trace = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        opt = OptimalPolicy(capacity=3)
        for t, page in enumerate(trace):
            opt.access(page, t, future_trace=trace[t + 1 :])
        # Textbook standard: 9 page faults
        self.assertEqual(opt.faults, 9)
        self.assertEqual(opt.hits, 11)


class TestModelsAndSimulation(unittest.TestCase):
    """Verifies that ML models and simulation engine execute without error."""

    def test_simulator_pipeline(self):
        gen = WorkloadGenerator(seed=123)
        train_trace, _, _ = gen.generate_shifted_workload(total_references=400, num_pages=20)
        eval_trace, shift_pt, _ = gen.generate_shifted_workload(total_references=400, num_pages=20)

        sim = Simulator(capacity=5, seed=123)
        log_reg, tree = sim.train_models(train_trace)

        results = sim.run_comparative_simulation(
            eval_trace=eval_trace,
            shift_point=shift_pt,
            log_reg_model=log_reg,
            tree_model=tree,
            record_explanations=True,
        )

        self.assertIn("summary_table", results)
        self.assertEqual(len(results["summary_table"]), 6)
        self.assertIsNotNone(results["calibration_metrics"])
        self.assertGreater(results["calibration_metrics"]["total_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()
