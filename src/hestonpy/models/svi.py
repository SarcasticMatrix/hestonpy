from scipy.optimize import NonlinearConstraint, basinhopping

from typing import Literal
import numpy as np
import warnings

class StochasticVolatilityInspired:
    """
    Implémente la paramétrisation SVI (Stochastic Volatility Inspired) dans sa formulation brute,
    telle que présentée dans le papier de Jim Gatheral.

    Cette classe permet de calculer la volatilité implicite totale selon la formulation SVI
    et d'effectuer une calibration aux données de marché.
    
    Attributs :
    -----------
    time_to_maturity : float
        Temps à l'échéance de l'option en années.
    """

    def __init__(self, time_to_maturity:float):

        self.time_to_maturity = time_to_maturity

    def raw_formulation(self, k, a:float, b:float, rho:float, m:float, sigma:float):
        """
        Calcule la variance totale implicite selon la formulation brute de SVI.

        Paramètres :
        ------------
        k : float
            Log-moneyness (log(strike / forward)).
        a : float
            Paramètre de décalage vertical.
        b : float
            Paramètre de courbure.
        rho : float
            Paramètre de corrélation (-1 < rho < 1).
        m : float
            Paramètre de translation horizontale.
        sigma : float
            Paramètre de volatilité d’échelle.

        Retour :
        --------
        float : Variance totale implicite associée au log-moneyness k.
        """
        return a + b * ( rho * (k-m) + np.sqrt((k-m)**2 + sigma**2) )
    
    def calibration(
            self, 
            strikes: np.array,
            market_ivs: np.array,
            forward: float,
            x0: list = [0.5, 0.5, 0.5, 0.5, 0.5],
            method: str = 'SLSQP'
            ):
        """
        Calibre le modèle SVI aux volatilités implicites de marché en minimisant l'erreur quadratique
        entre la variance totale implicite du modèle et celle du marché.

        Paramètres :
        ------------
        strikes : np.array
            Tableau des prix d'exercice des options.
        market_ivs : np.array
            Tableau des volatilités implicites du marché.
        forward : float
            Prix à terme du sous-jacent. Souvent np.exp(r * time_to_mat) * spot
        x0 : list, optionnel
            Valeurs initiales des paramètres SVI (respectivement a, b, rho, m, sigma, par défaut [0.5, 0.5, 0.5, 0.5, 0.5]).
        method : str, optionnel
            Algorithme d'optimisation utilisé (par défaut 'SLSQP').

        Retour :
        --------
        tuple (dict, np.array) :
            - Dictionnaire des paramètres SVI calibrés { "a": ..., "b": ..., "rho": ..., "m": ..., "sigma": ... }.
            - Tableau des volatilités implicites du modèle après calibration.
        """
        
        market_total_implied_variance = market_ivs**2 * self.time_to_maturity
        def cost_function(params):   
            a, b, rho, m, sigma = params
            formulation_params = {
                "a":a,
                "b":b,
                "rho":rho,
                "m":m,
                "sigma":sigma
            }

            model_total_implied_variance = self.raw_formulation(np.log(strikes/forward), **formulation_params)
            return np.sum((model_total_implied_variance - market_total_implied_variance) ** 2)
        
        # Bounds of parameters
        bounds = [
            (-1, 1),    
            (1e-3, 5),  
            (-0.999, 0.999), 
            (-2, 2),    
            (1e-3, 5)   
        ]

        # Constraints
        con = lambda x: x[0] + x[1] * x[4] * np.sqrt(1 - x[2]**2)
        minimizer_kwargs = {
                "method": method,
                "bounds": bounds,
                "constraints": NonlinearConstraint(con, lb=0, ub=np.inf)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = basinhopping(
                cost_function, 
                x0=x0,
                niter=5000,
                stepsize=0.5,
                niter_success=10,
                minimizer_kwargs=minimizer_kwargs,
            )
        print(result.message, result.success)

        calibrated_params = {
                "a": result.x[0],
                "b": result.x[1],
                "rho": result.x[2],
                "m": result.x[3],
                "sigma": result.x[4]
        }
        calibrated_ivs = np.sqrt(self.raw_formulation(np.log(strikes/forward), **calibrated_params) / self.time_to_maturity)

        return calibrated_params, calibrated_ivs

