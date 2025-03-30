"""
All the functions related to implied volatility surface
"""

from hestonpy.models.blackScholes import BlackScholes
from typing import Literal
import numpy as np

def dichotomie(
        market_price,
        price_function,
        error: float = 10**(-6),
        vol_inf: float = 10**(-3),
        vol_sup: float = 1
    ):
    """
    price_function should be only a function of the volatility
    Note that the price_function is always a croissant function of the volatility
    """
    target_function = lambda volatility: price_function(volatility) - market_price

    while vol_sup - vol_inf > error:
        vol_mid = (vol_inf + vol_sup)/2
        if target_function(vol_inf) * target_function(vol_mid) < 0:
            vol_sup = vol_mid
        else:
            vol_inf = vol_mid
            
    return vol_mid

def newton_raphson(
        market_price,
        price_function,
        vega_function,
        initial_guess: float = 0.2,
        tolerance: float = 10**(-6),
        max_iterations: int = 100
    ):
    """
    Implements the Newton-Raphson method to find implied volatility.
    price_function should be only a function of volatility that returns the option price.
    vega_function should be a function of volatility that returns Vega.
    """
    volatility = initial_guess
    for _ in range(max_iterations):
        price_diff = price_function(volatility) - market_price
        vega = vega_function(volatility)
        
        if abs(price_diff) < tolerance:
            return volatility
        
        if vega == 0:  # Avoid division by zero
            break
        
        volatility -= price_diff / vega
    
    return volatility

def reverse_blackScholes(
        price: float,
        strike: float,
        time_to_maturity: float,
        bs: BlackScholes,
        flag_option: Literal['call','put'] = 'call',
        method: Literal['dichotomie', 'newton_raphson'] = 'dichotomie'
):
    """
    Reverse the Black-Scholes formula, compute the implied volatility from market price.
    bs should be already initialized with the right strike and maturity.
    """
    if flag_option == 'call':
        bs_price = lambda volatility: bs.call_price(strike=strike, time_to_maturity=time_to_maturity, volatility=volatility)
    else:
        bs_price = lambda volatility: bs.put_price(strike=strike, time_to_maturity=time_to_maturity, volatility=volatility)
    
    vega_function = lambda volatility: bs.vega(strike=strike, time_to_maturity=time_to_maturity, volatility=volatility)
    
    if method == 'dichotomie':
        iv = dichotomie(market_price=price, price_function=bs_price)
    elif method == 'newton_raphson':
        iv = newton_raphson(market_price=price, price_function=bs_price, vega_function=vega_function)
    else:
        raise ValueError("Invalid method. Choose either 'dichotomie' or 'newton_raphson'.")
    
    return iv

def compute_smile(
        prices: float,
        strikes: float,
        time_to_maturity: float,
        bs: BlackScholes,
        flag_option: Literal['call','put'],
        method: Literal['dichotomie', 'newton_raphson'] = 'dichotomie'
    ):

    ivs = []
    for (price, strike) in zip(prices, strikes):
        iv = reverse_blackScholes(
            price=price, 
            strike=strike, 
            bs=bs, 
            time_to_maturity=time_to_maturity, 
            flag_option='call', 
            method=method
        )
        ivs.append(iv)

    return np.array(ivs)


class CustomStep:
    """
    Par défaut, basinhopping utilise un saut uniforme sur toutes les dimensions, ce qui peut être sous-optimal. 
    Un saut gaussien mieux peut être plus adapté à l’échelle de chaque paramètre :
    """

    def __init__(self, scale):
        self.scale = scale
    def __call__(self, x):
        return x + np.random.normal(scale=self.scale, size=len(x))  # Sauts gaussiens

feller = lambda x: 4 * x[0] * x[1] / x[2]**2

# Cost function and power, relative_errors parameters
def generate_difference_function(power:str, relative_errors:bool):
    if not relative_errors:
        difference = lambda market_prices, model_prices : market_prices - model_prices
    else:
        difference = lambda market_prices, model_prices : (market_prices - model_prices) / market_prices
            
    if power == 'mae':
        return lambda market_prices, model_prices : np.sum(np.abs(difference(market_prices, model_prices)))
    elif power == 'rmse':
        return lambda market_prices, model_prices : np.sum(np.sqrt(difference(market_prices, model_prices)**2))
    elif power == 'mse':
        return lambda market_prices, model_prices : np.sum(difference(market_prices, model_prices)**2)
    else: 
        raise ValueError("Invalid power. Choose either 'rmse', 'mae', or 'mse'.")
    

def get_parameters(model_type:str, params:list):
    
    if model_type == "Heston":
        kappa, theta, sigma, rho = params
        function_params = {
            "kappa": kappa,
            "theta": theta,
            "drift_emm": 0, 
            "sigma": sigma,
            "rho": rho,
        }
            
    elif model_type == "Bates":
        kappa, theta, sigma, rho, lambda_jump, mu_J, sigma_J = params
        function_params = {
            "kappa": kappa,
            "theta": theta,
            "drift_emm": 0,
            "sigma": sigma,
            "rho": rho,
            "lambda_jump": lambda_jump,
            "mu_J": mu_J,
            "sigma_J": sigma_J,
        }
    return function_params

def set_bounds(model_type, guess_correlation_sign, initial_guess):
        
    # Bounds of parameters
    bounds = [
        (1e-3, 10),  # kappa 
        (1e-3, 3),   # theta
        (1e-3, 6),   # sigma
    ]                # rho
    if guess_correlation_sign == 'positive':
        bounds.append((0.0,1.0))
        if initial_guess[-1] < 0:
            initial_guess[-1] = - initial_guess[-1]
    elif guess_correlation_sign == 'negative':
        bounds.append((-1.0, 0.0))
        if initial_guess[-1] > 0:
            initial_guess[-1] = - initial_guess[-1]
    elif guess_correlation_sign == 'unknown':
        bounds.append((-1.0,1.0))

    if model_type == "Heston":
        return bounds
    else:
        jump_parameters_bounds = [(0.01, 5.0), (-0.2, 0.2), (0.05, 0.5)] # lambda_jump, mu_J, sigma_J,
        bounds = bounds + jump_parameters_bounds
        return bounds

def get_calibrated_params(calibrated_params, result, vol_initial, model_type):

    if model_type == 'Heston':
        calibrated_params = {
            "vol_initial": vol_initial, 
            "kappa": result.x[0],
            "theta": result.x[1],
            "sigma": result.x[2],
            "rho": result.x[3],
            "drift_emm": 0,
        }
        print(f"Calibrated parameters: v0={vol_initial:.3f} | kappa={result.x[0]:.3f} | theta={result.x[1]:.3f} | sigma={result.x[2]:.3f} | rho={result.x[3]:.3f}\n")

    else: 
        calibrated_params = {
            "vol_initial": vol_initial, 
            "kappa": result.x[0],
            "theta": result.x[1],
            "drift_emm": 0,
            "sigma": result.x[2],
            "rho": result.x[3],
            "lambda_jump": result.x[4],
            "mu_J": result.x[5],
            "sigma_J": result.x[6],
        }
        print(f"Calibrated parameters: v0={vol_initial:.3f} | kappa={result.x[0]:.3f} | theta={result.x[1]:.3f} | sigma={result.x[2]:.3f} | rho={result.x[3]:.3f}  | lambda_jump={result.x[4]:.3f}  | mu_J={result.x[5]:.3f}  | sigma_J={result.x[6]:.3f}\n")

    return calibrated_params