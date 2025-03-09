from scipy.optimize import NonlinearConstraint, basinhopping
import warnings
import numpy as np

class StochasticVolatilityInspired:

    def __init__(self, time_to_maturity:float):

        self.time_to_maturity = time_to_maturity

    def raw_formulation(self, k, a:float, b:float, rho:float, m:float, sigma:float):

        return a + b * ( rho * (k-m) + np.sqrt((k-m)**2 + sigma**2) )
    
    def calibration(
            self, 
            strikes: np.array,
            market_ivs: np.array,
            forward: float,
            x0: list = [0.5, 0.5, 0.5, 0.5, 0.5],
            method: str = 'SLSQP'
            ):
        
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

