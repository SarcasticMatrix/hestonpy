"""
    Smoke tests for BlackScholes class 
"""
import unittest
import numpy as np
from hestonpy.models.blackScholes import BlackScholes

class TestBlackScholesSmoke(unittest.TestCase):
    def setUp(self):
        """Initialisation d'une instance de BlackScholes pour les tests."""
        self.model = BlackScholes(spot=100, r=0.05, mu=0.05, volatility=0.2, seed=42)

    def test_instance_creation(self):
        """Vérifie que l'instance est bien créée sans erreur."""
        self.assertIsInstance(self.model, BlackScholes)

    def test_simulate(self):
        """Vérifie que la simulation s'exécute sans erreur et retourne un tableau NumPy."""
        result = self.model.simulate(time_to_maturity=1, scheme="euler", nbr_points=10, nbr_simulations=5)
        self.assertIsInstance(result, np.ndarray)

    def test_call_price(self):
        """Vérifie que la fonction de pricing d'option call ne génère pas d'erreur."""
        price = self.model.call_price(strike=100, time_to_maturity=1)
        self.assertIsInstance(price, float)

    def test_put_price(self):
        """Vérifie que la fonction de pricing d'option put ne génère pas d'erreur."""
        price = self.model.put_price(strike=100, time_to_maturity=1)
        self.assertIsInstance(price, float)

    def test_delta(self):
        """Vérifie que le calcul du delta fonctionne sans erreur."""
        delta = self.model.delta(strike=100, flag_option="call", time_to_maturity=1)
        self.assertIsInstance(delta, float)

    def test_gamma(self):
        """Vérifie que le calcul du gamma fonctionne sans erreur."""
        gamma = self.model.gamma(strike=100, time_to_maturity=1)
        self.assertIsInstance(gamma, float)

    def test_delta_hedging(self):
        """Vérifie que delta_hedging fonctionne sans erreur."""
        portfolio, S = self.model.delta_hedging(strike=100, time_to_maturity=1, flag_option='call', hedging_volatility=self.model.volatility)
        self.assertIsInstance(portfolio, np.ndarray)
        self.assertIsInstance(S, np.ndarray)

    def test_volatility_arbitrage(self):
        """Vérifie que volatility_arbitrage fonctionne sans erreur."""
        portfolio, S = self.model.volatility_arbitrage(strike=100, time_to_maturity=1, flag_option='call', hedging_volatility=self.model.volatility)
        self.assertIsInstance(portfolio, np.ndarray)
        self.assertIsInstance(S, np.ndarray)

if __name__ == "__main__":
    unittest.main()
