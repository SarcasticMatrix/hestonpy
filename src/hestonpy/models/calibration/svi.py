from scipy.optimize import NonlinearConstraint, basinhopping

from typing import Literal
import numpy as np
import warnings

class StochasticVolatilityInspired:
    """
    Implements the SVI (Stochastic Volatility Inspired) parameterization in its raw formulation,
    as presented in Jim Gatheral's paper.

    This class allows for the calculation of the total implied volatility according to the SVI formulation
    and performs calibration to market data.

    :param float time_to_maturity: Time to maturity of the option in years.
    """

    def __init__(self, time_to_maturity:float):
        """
        Initializes the StochasticVolatilityInspired object.

        :param float time_to_maturity: Time to maturity of the option in years.
        """
        self.time_to_maturity = time_to_maturity

    def raw_formulation(self, k, a:float, b:float, rho:float, m:float, sigma:float):
        """
        Computes the total implied variance according to the raw SVI formulation.

        :param float k: Log-moneyness (log(strike / forward)).
        :param float a: Vertical shift parameter.
        :param float b: Curvature parameter.
        :param float rho: Correlation parameter (-1 < rho < 1).
        :param float m: Horizontal translation parameter.
        :param float sigma: Scale volatility parameter.

        :returns: Total implied variance associated with log-moneyness k.
        :rtype: float
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
        Calibrates the SVI model to market implied volatilities by minimizing the squared error
        between the model's total implied variance and the market's total implied variance.

        :param np.array strikes: Array of option strike prices.
        :param np.array market_ivs: Array of market implied volatilities.
        :param float forward: Forward price of the underlying asset. Often np.exp(r * time_to_mat) * spot.
        :param list x0: Initial values of the SVI parameters (a, b, rho, m, sigma, respectively). Default is [0.5, 0.5, 0.5, 0.5, 0.5].
        :param str method: Optimization algorithm used. Default is 'SLSQP'.

        :returns: Tuple containing a dictionary of calibrated SVI parameters and an array of model implied volatilities after calibration.
        :rtype: tuple (dict, np.array)
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

