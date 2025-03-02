import numpy as np
from hestonpy.models.blackScholes import BlackScholes
from typing import Literal

def volatility_dichotomie(
        market_price,
        price_function,
        error: float = 10**(-3),
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



def reverse_blackScholes(
        price: float,
        strike: float,
        T: float,
        bs: BlackScholes,
        flag_option: Literal['call','put']
):
    """
    Reverse the blackScholes formula, compute the implied volatility from market price.
    bs should be already with the right stirke and maturity
    """

    bs_price = lambda volatility: bs.call_price(K=strike, T=T, volatility=volatility)
