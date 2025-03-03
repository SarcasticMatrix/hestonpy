import unittest
import numpy as np
from hestonpy.models.heston import Heston

class TestHestonSmoke(unittest.TestCase):
    def setUp(self):
        """Initialisation d'une instance de Heston pour les tests."""
        self.model = Heston(
            spot=100, vol_initial=0.04, r=0.05, kappa=2.0, theta=0.04,
            drift_emm=0.0, sigma=0.1, rho=-0.7, premium_volatility_risk=0.0, seed=42
        )

    def test_instance_creation(self):
        """Vérifie que l'instance est bien créée sans erreur."""
        self.assertIsInstance(self.model, Heston)

    def test_simulate(self):
        """Vérifie que la simulation s'exécute sans erreur et retourne un tuple de tableaux NumPy."""
        S, V, null_variance = self.model.simulate(time_to_maturity=1, scheme="euler", nbr_points=10, nbr_simulations=5)
        self.assertIsInstance(S, np.ndarray)
        self.assertIsInstance(V, np.ndarray)
        self.assertIsInstance(null_variance, int)
    
    def test_monte_carlo_price(self):
        """Vérifie que la fonction de pricing Monte Carlo retourne un résultat correct."""
        result = self.model.monte_carlo_price(strike=100, time_to_maturity=1, scheme="euler")
        self.assertIsInstance(result.price, float)
    
    def test_fourier_transform_price(self):
        """Vérifie que la méthode de pricing par transformée de Fourier fonctionne sans erreur."""
        price, error = self.model.fourier_transform_price(strike=100, time_to_maturity=1, error_boolean=True)
        self.assertIsInstance(price, float)
        self.assertIsInstance(error, float)
    
    def test_carr_madan_price(self):
        """Vérifie que la méthode Carr-Madan retourne un prix et une erreur."""
        price, error = self.model.carr_madan_price(strike=100, time_to_maturity=1, error_boolean=True)
        self.assertIsInstance(price, float)
        self.assertIsInstance(error, float)
    
    def test_call_delta(self):
        """Vérifie que le calcul du delta fonctionne sans erreur."""
        delta = self.model.call_delta(strike=100, time_to_maturity=1, s=100, v=0.04)
        self.assertIsInstance(delta, float)
    
    def test_call_vega(self):
        """Vérifie que le calcul du vega fonctionne sans erreur."""
        vega = self.model.call_vega(strike=100, time_to_maturity=1, s=100, v=0.04)
        self.assertIsInstance(vega, float)
    
if __name__ == "__main__":
    unittest.main()
