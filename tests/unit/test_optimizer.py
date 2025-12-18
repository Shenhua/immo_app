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
        df_data = {"Cash-Flow Net d'Impôt": [1000.0] * 20, "Charges Déductibles": [-100]*20, "Loyers Bruts": [1200]*20}
        self.mock_df = pd.DataFrame(df_data)
        self.mock_bilan = {"tri_annuel": 5.0, "enrichissement_net": 50000.0, "liquidation_nette": 100000.0, "dscr_y1": 1.2}
        self.mock_simulator.simulate.return_value = (self.mock_df, self.mock_bilan)
        
        self.mock_scorer.qualite_weight = 0.5
        self.mock_scorer.weights = {
            "enrich_net": 0.30, "irr": 0.25, "cf_proximity": 0.20, "dscr": 0.15, "cap_eff": 0.10
        }
        
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
        # Need to provide actual details (not empty) for evaluation
        mock_details = [
            {"nom_bien": "B0", "credit_final": 80000, "taux_pret": 3.5, 
             "duree_pret": 20, "assurance_ann_pct": 0.36, 
             "qual_score_bien": 70.0, "cout_total": 100000}
        ]
        self.mock_allocator.allocate.return_value = (True, mock_details, 0.0, 10000.0)
        self.mock_allocator.mode_cf = "min"
        
        ind = Individual(self.sample_bricks[:2])
        self.optimizer._evaluate(ind, 50000, 0, 100, 20)
        
        self.assertTrue(ind.is_valid)
        self.assertGreater(ind.fitness, 0.0)
        # Note: key changed from cf_year_1_monthly to cf_monthly_y1 after refactoring
        self.assertIn("cf_monthly_y1", ind.stats)

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
