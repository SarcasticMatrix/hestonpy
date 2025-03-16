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
    ):
    """
    price_function should be only a function of the volatility
    Note that the price_function is always a croissant function of the volatility
    """
    nbrPoints = int(1/error)+1
    interval = np.linspace(start=0, stop=1, num=nbrPoints)
    index_inf = 0
    index_sup = nbrPoints-1

    target_function = lambda volatility: price_function(volatility) - market_price

    while (index_sup - index_inf) > 1:

        index_mid = (index_inf + index_sup) // 2

        if target_function(interval[index_mid]) > 0:
            index_sup = index_mid

        else:
            index_inf = index_mid

    return (interval[index_inf] + interval[index_sup]) / 2

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
