import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.services.optimizer import GeneticOptimizer, Individual

class TestGeneticOptimizer(unittest.TestCase):
    def setUp(self):
        # Mocks
        self.mock_allocator = MagicMock()
        self.mock_simulator = MagicMock()
        self.mock_scorer = MagicMock()
        
        # Setup common allocator return: ok, details, cf_final, apport_used
        self.mock_allocator.allocate.return_value = (True, [], 0.0, 10000.0)
        
        # Setup common simulator return: (df, bilan)
        df_data = {"Cash Flow": [1000.0] * 20, "Charges Déductibles": [-100]*20, "Loyers Bruts": [1200]*20}
        self.mock_df = pd.DataFrame(df_data)
        self.mock_bilan = {"tri_global": 5.0, "enrichissement_net": 50000.0, "dscr_y1": 1.2}
        self.mock_simulator.simulate.return_value = (self.mock_df, self.mock_bilan)
        
        self.mock_scorer.qualite_weight = 0.5
        
        self.optimizer = GeneticOptimizer(
            population_size=10,
            generations=2,
            allocator=self.mock_allocator,
            simulator=self.mock_simulator,
            scorer=self.mock_scorer
        )
        
        self.sample_bricks = [
            {
                "nom_bien": f"B{i}", 
                "apport_min": 10000.0,
                "loyer_mensuel_initial": 500.0 + (i * 10),
                "cout_total": 100000.0 + (i * 1000),
                "qual_score_bien": 10.0 * i
            } for i in range(10)
        ]

    def test_initialization(self):
        self.assertEqual(self.optimizer.pop_size, 10)
        self.assertEqual(self.optimizer.generations, 2)

    def test_generate_random_individual(self):
        ind = self.optimizer._generate_random_individual(
            self.sample_bricks, budget=25000.0, target_cf=0, tolerance=0
        )
        self.assertIsInstance(ind, Individual)
        # Should fit <= 2 bricks (20k <= 25k)
        total_apport = sum(b["apport_min"] for b in ind.bricks)
        self.assertLessEqual(total_apport, 25000.0)

    def test_evaluate_success(self):
        ind = Individual(self.sample_bricks[:2])
        self.optimizer._evaluate(ind, 50000, 0, 100, 20)
        
        self.assertTrue(ind.is_valid)
        self.assertGreater(ind.fitness, 0.0)
        self.assertIn("cf_year_1_monthly", ind.stats)
        self.assertEqual(ind.stats["cf_year_1_monthly"], 1000.0/12)

    def test_evaluate_allocation_fail(self):
        # Allocator returns False
        self.mock_allocator.allocate.return_value = (False, [], 0.0, 0.0)
        ind = Individual(self.sample_bricks[:2])
        self.optimizer._evaluate(ind, 50000, 0, 100, 20)
        
        self.assertFalse(ind.is_valid)
        self.assertEqual(ind.fitness, 0.1)

    def test_evaluate_simulation_exception(self):
        self.mock_simulator.simulate.side_effect = Exception("Sim Fail")
        ind = Individual(self.sample_bricks[:2])
        self.optimizer._evaluate(ind, 50000, 0, 100, 20)
        
        self.assertFalse(ind.is_valid)
        self.assertEqual(ind.fitness, 0.0)

    def test_evolve_loop(self):
        # Should run without error and return list of strategies
        results = self.optimizer.evolve(
            self.sample_bricks, 
            budget=50000, 
            target_cf=0, 
            tolerance=100
        )
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= 10) # pop size
        if results:
            self.assertIn("fitness", results[0])
            self.assertIn("details", results[0])

if __name__ == "__main__":
    unittest.main()
