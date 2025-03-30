import numpy as np
from numpy import random
from scipy.integrate import quad, quad_vec
from tqdm import tqdm
from typing import Literal

import matplotlib.pyplot as plt
from collections import namedtuple


class Bates:
    """
    stochastic volatility jump (SVJ) model is suggested by Bates
    """

    def __init__(
        self,
        spot,
        vol_initial,
        r,
        kappa,
        theta,
        drift_emm,
        sigma,
        rho,
        lambda_jump,
        mu_J,
        sigma_J,
        seed=42,
    ):

        # Simulation parameters
        self.spot = spot  # spot price
        self.vol_initial = vol_initial  # initial variance

        # Model parameters
        self.kappa = kappa  # mean reversion speed
        self.theta = theta  # long term variance
        self.sigma = sigma  # vol of variance
        self.rho = rho  # correlation
        self.drift_emm = drift_emm  # lambda from P to martingale measure Q (Equivalent Martingale Measure)

        # Jump parameters
        self.lambda_jump = lambda_jump
        self.mu_J = mu_J
        self.sigma_J = sigma_J

        # World parameters
        self.r = r  # interest rate

        self.seed = seed  # random seed

    def characteristic(self, j: int, **kwargs) -> float:
        """
        Extends the characteristic function to include jumps in the Heston model.
        """
        vol_initial = kwargs.get("vol_initial", self.vol_initial)

        # Model parameters
        kappa = kwargs.get("kappa", self.kappa)
        theta = kwargs.get("theta", self.theta)
        sigma = kwargs.get("sigma", self.sigma)
        rho = kwargs.get("rho", self.rho)
        drift_emm = kwargs.get("drift_emm", self.drift_emm)

        # Jump parameters
        lambda_jump = kwargs.get("lambda_jump", self.lambda_jump)
        mu_J = kwargs.get("mu_J", self.mu_J)
        sigma_J = kwargs.get("sigma_J", self.sigma_J)

        if j == 1:
            uj = 1 / 2
            bj = kappa + drift_emm - rho * sigma
        elif j == 2:
            uj = -1 / 2
            bj = kappa + drift_emm
        else:
            print("Argument j (int) must be 1 or 2")
            return 0
        a = kappa * theta / sigma**2

        dj = lambda u: np.sqrt((rho * sigma * u * 1j - bj) ** 2 - sigma**2 * (2 * uj * u * 1j - u**2))
        gj = lambda u: (rho * sigma * u * 1j - bj - dj(u)) / (rho * sigma * u * 1j - bj + dj(u))

        Cj = lambda tau, u: self.r * u * tau * 1j + a * (
            (bj - rho * sigma * u * 1j + dj(u)) * tau - 2 * np.log((1 - gj(u) * np.exp(dj(u) * tau)) / (1 - gj(u)))
        )
        Dj = lambda tau, u: (bj - rho * sigma * u * 1j + dj(u)) / sigma**2 * (1 - np.exp(dj(u) * tau)) / (1 - gj(u) * np.exp(dj(u) * tau))

        # Jump component
        char_jump = lambda u, tau: np.exp(lambda_jump * tau * (np.exp(1j * u * mu_J - 0.5 * sigma_J**2 * u**2) - 1))

        return lambda x, v, time_to_maturity, u: (
            np.exp(Cj(time_to_maturity, u) + Dj(time_to_maturity, u) * v + u * x * 1j) * char_jump(u, time_to_maturity)
        )

    def call_price(
            self, 
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
            **kwargs
        ):
        
        price = self.carr_madan_price(
            s=s, 
            v=v,
            strike=strike, 
            time_to_maturity=time_to_maturity,
            **kwargs
        )
        return price

    
    def carr_madan_price(
            self, 
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
            error_boolean: bool = False,
            **kwargs
        ):
        """
        Computes the price of a European call option using the Carr-Madan Fourier pricing method.

        This method employs the Carr-Madan approach, leveraging the characteristic function to calculate
        the option price.

        Returns:
        - price (float): The calculated option price.
        - error (float): The error associated with the option price calculation.
        """

        if s is None:
            s = self.spot
        x = np.log(s)
        if v is None:
            v = kwargs.get("vol_initial", self.vol_initial)  # Initial variance
        alpha = 0.3

        price_hat = (
            lambda u: np.exp(-self.r * time_to_maturity)
            / (alpha**2 + alpha - u**2 + u * (2 * alpha + 1) * 1j)
            * self.characteristic(j=2, **kwargs)(x, v, time_to_maturity, u - (alpha + 1) * 1j)
        )

        integrand = lambda u: np.exp(-u * np.log(strike) * 1j) * price_hat(u)

        price = np.real(
            np.exp(-alpha * np.log(strike)) / np.pi * quad_vec(f=integrand, a=0, b=1000)[0]
        )

        if error_boolean:
            error = (
                np.exp(-alpha * np.log(strike)) / np.pi * quad_vec(f=integrand, a=0, b=1000)[1]
            )
            return price, error
        else: 
            return price   
   