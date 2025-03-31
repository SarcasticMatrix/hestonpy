import numpy as np
from numpy import random
from scipy.integrate import quad, quad_vec
from tqdm import tqdm
from typing import Literal

import matplotlib.pyplot as plt
from collections import namedtuple


class Heston:
    """
    Heston model for option pricing with stochastic volatility.

    This class implements the Heston model for pricing European options with stochastic volatility.
    It includes methods for simulation, pricing, and hedging.

    You can initialize the Heston model parameters under the historical or risk neutral dynamics as you want.

    :param float spot: Current price of the underlying asset.
    :param float vol_initial: Initial variance of the underlying asset.
    :param float r: Risk-free interest rate.
    :param float kappa: Mean reversion speed of the variance.
    :param float theta: Long-term mean of the variance.
    :param float drift_emm: Market price of risk (lambda) for the variance process. Usually set as 0
    :param float sigma: Volatility of the variance (vol of vol).
    :param float rho: Correlation between the asset price and its variance.
    :param int seed: Random seed for reproducibility.
    """

    def __init__(self, spot:float, vol_initial:float, r:float, kappa:float, theta:float, drift_emm:float, sigma:float, rho:float, seed:int=42):

        # Simulation parameters
        self.spot = spot  # spot price
        self.vol_initial = vol_initial  # initial variance

        # Model parameters
        self.kappa = kappa  # mean reversion speed
        self.theta = theta  # long term variance
        self.sigma = sigma  # vol of variance
        self.rho = rho  # correlation
        self.drift_emm = drift_emm  # lambda from P to martingale measure Q (Equivalent Martingale Measure)

        # World parameters
        self.r = r  # interest rate

        self.seed = seed  # random seed

    def simulate(
            self,
            time_to_maturity: float = 1,
            scheme: str = Literal["euler", "milstein"],
            nbr_points: int = 100,
            nbr_simulations: int = 1000,
        ) -> tuple:
        """
        Simulate asset price and variance paths using the Heston model.

        This method simulates the paths of the underlying asset price and its variance
        using either the Euler or Milstein discretization scheme.

        :param float time_to_maturity: Time to maturity of the option in years.
        :param str scheme: Discretization scheme to use ('euler' or 'milstein').
        :param int nbr_points: Number of time points in each simulation.
        :param int nbr_simulations: Number of simulations to run.

        :returns: Simulated asset prices, variances, and count of null variances.
        :rtype: tuple
        """

        random.seed(self.seed)

        dt = time_to_maturity / nbr_points
        S = np.zeros((nbr_simulations, nbr_points + 1))
        V = np.zeros((nbr_simulations, nbr_points + 1))
        S[:, 0] = self.spot
        V[:, 0] = self.vol_initial

        null_variance = 0

        for i in range(1, nbr_points + 1):

            # Apply reflection scheme
            if np.any(V[:, i - 1] < 0):
                V[:, i - 1] = np.abs(V[:, i - 1])

            if np.any(V[:, i - 1] == 0):
                null_variance += np.sum(V[i - 1, :] == 0)

            # Brownian motion
            N1 = np.random.normal(loc=0, scale=1, size=nbr_simulations)
            N2 = np.random.normal(loc=0, scale=1, size=nbr_simulations)
            ZV = N1 * np.sqrt(dt)
            ZS = (self.rho * N1 + np.sqrt(1 - self.rho**2) * N2) * np.sqrt(dt)

            # Update the processes
            # S[:, i] = S[:, i-1] + self.r * S[:, i-1] * dt + np.sqrt(V[:, i-1]) * S[:, i-1] * ZS
            S[:, i] = (
                S[:, i - 1]
                + (self.r + self.drift_emm * np.sqrt(V[:, i - 1])) * S[:, i - 1] * dt
                + np.sqrt(V[:, i - 1]) * S[:, i - 1] * ZS
            )

            V[:, i] = (
                V[:, i - 1]
                + (
                    self.kappa * (self.theta - V[:, i - 1])
                    - self.drift_emm * V[:, i - 1]
                )
                * dt
                + self.sigma * np.sqrt(V[:, i - 1]) * ZV
            )
            if scheme == "milstein":
                S[:, i] += 1 / 2 * V[:, i - 1] * S[:, i - 1] * (ZS**2 - dt)
                # S[:, i] += 1/4 * S[:, i-1]**2 * (ZS**2 - dt)
                V[:, i] += 1 / 4 * self.sigma**2 * (ZV**2 - dt)

        if nbr_simulations == 1:
            S = S.flatten()
            V = V.flatten()

        return S, V, null_variance

    def plot_simulation(
        self,
        time_to_maturity: float = 1,
        scheme: str = Literal["euler", "milstein"],
        nbr_points: int = 100,
    ) -> tuple:
        """
        Plot a single simulation of the asset price and variance paths.

        This method simulates and plots the paths of the underlying asset price and its variance
        using the specified discretization scheme.

        :param float time_to_maturity: Time to maturity of the option in years.
        :param str scheme: Discretization scheme to use ('euler' or 'milstein').
        :param int nbr_points: Number of time points in the simulation.

        :returns: Simulated asset prices and variances.
        :rtype: tuple
        """
 
        S, V, _ = self.simulate(
            time_to_maturity=time_to_maturity, 
            scheme=scheme, 
            nbr_points=nbr_points, 
            nbr_simulations=1
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,8))

        ax1.plot(
            np.linspace(0, 1, nbr_points + 1), S, label="Risky asset", color="blue", linewidth=1
        )
        ax1.set_ylabel("Value [$]", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8,)

        ax2.plot(np.linspace(0, 1, nbr_points + 1),np.sqrt(V),label="Volatility",color="orange",linewidth=1,)
        ax2.axhline(y=np.sqrt(self.theta),label=r"$\sqrt{\theta}$",linestyle="--",color="black",)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Instantaneous volatility [%]", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8,)

        fig.suptitle(f"Heston Model Simulation with {scheme} scheme", fontsize=16)
        plt.tight_layout()
        plt.show()

        return S, V
    
    def monte_carlo_price(
        self,
        strike: float,
        time_to_maturity: float,
        scheme: str = Literal["euler", "milstein"],
        nbr_points: int = 100,
        nbr_simulations: int = 1000,
    ):
        """
        Price a European option using Monte Carlo simulation.

        This method prices a European option using Monte Carlo simulation with the Heston model.

        :param float strike: Strike price of the option.
        :param float time_to_maturity: Time to maturity of the option in years.
        :param str scheme: Discretization scheme to use ('euler' or 'milstein').
        :param int nbr_points: Number of time points in each simulation.
        :param int nbr_simulations: Number of simulations to run.

        :returns: Option price and confidence interval.
        :rtype: namedtuple
        """
        
        S, _, null_variance = self.simulate(
            time_to_maturity=time_to_maturity, 
            scheme=scheme, 
            nbr_points=nbr_points, 
            nbr_simulations=nbr_simulations
        )
        print(
            f"Variance has been null {null_variance} times over the {nbr_points*nbr_simulations} iterations ({round(null_variance/(nbr_points*nbr_simulations)*100,2)}%) "
        )

        ST = S[:, -1]
        payoff = np.maximum(ST - strike, 0)
        discounted_payoff = np.exp(-self.r * time_to_maturity) * payoff

        price = np.mean(discounted_payoff)
        standard_deviation = np.std(discounted_payoff, ddof=1) / np.sqrt(nbr_simulations)
        infimum = price - 1.96 * np.sqrt(standard_deviation / nbr_simulations)
        supremum = price + 1.96 * np.sqrt(standard_deviation / nbr_simulations)

        Result = namedtuple("Results", "price std infinum supremum")
        return Result(
            price, standard_deviation, infimum, supremum
        )

    def characteristic(self, j: int, **kwargs):
        """
        Creates the characteristic function Psi_j(x, v, t; u) for a given (x, v, t).

        This function returns the characteristic function based on the index provided.

        :param int j: Index of the characteristic function (must be 1 or 2).

        :returns: The characteristic function.
        :rtype: callable

        :raises ValueError: If the index j is not 1 or 2.
        """

        vol_initial = kwargs.get("vol_initial", self.vol_initial)  # Initial variance

        # Model parameters
        kappa = kwargs.get("kappa", self.kappa)
        theta = kwargs.get("theta", self.theta)
        sigma = kwargs.get("sigma", self.sigma)
        rho = kwargs.get("rho", self.rho)
        drift_emm = kwargs.get("drift_emm", self.drift_emm)

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

        dj = lambda u: np.sqrt(
            (rho * sigma * u * 1j - bj) ** 2
            - sigma**2 * (2 * uj * u * 1j - u**2)
        )
        gj = lambda u: (rho * sigma * u * 1j - bj - dj(u)) / (
            rho * sigma * u * 1j - bj + dj(u)
        )

        Cj = lambda tau, u: self.r * u * tau * 1j + a * (
            (bj - rho * sigma * u * 1j + dj(u)) * tau
            - 2 * np.log((1 - gj(u) * np.exp(dj(u) * tau)) / (1 - gj(u)))
        )
        Dj = (
            lambda tau, u: (bj - rho * sigma * u * 1j + dj(u))
            / sigma**2
            * (1 - np.exp(dj(u) * tau))
            / (1 - gj(u) * np.exp(dj(u) * tau))
        )

        return lambda x, v, time_to_maturity, u: np.exp(
            Cj(time_to_maturity, u) + Dj(time_to_maturity, u) * v + u * x * 1j
        )

    def fourier_transform_price(
            self, 
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
            error_boolean: bool = False,
            **kwargs
        ):
        """
        Price a European option using the Fourier transform method.

        This method prices a European option using the Fourier transform method with the Heston model.

        :param np.array strike: Strike prices of the options.
        :param np.array time_to_maturity: Times to maturity of the options.
        :param np.array s: Current prices of the underlying asset.
        :param np.array v: Initial variances of the underlying asset.
        :param bool error_boolean: Whether to return the error of the price calculation.

        :returns: Option prices and errors (if error_boolean is True).
        :rtype: float or tuple
        """

        if s is None:
            s = self.spot
        x = np.log(s)
        if v is None:
            v = kwargs.get("vol_initial", self.vol_initial)  # Initial variance
        
        psi1 = self.characteristic(j=1, **kwargs)
        integrand1 = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi1(x, v, time_to_maturity, u)) / (u * 1j)
        )
        Q1 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand1, a=0, b=1000)[0]
        if error_boolean:
            error1 = 1 / np.pi * quad_vec(f=integrand1, a=0, b=1000)[1]

        psi2 = self.characteristic(j=2, **kwargs)
        integrand2 = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi2(x, v, time_to_maturity, u)) / (u * 1j)
        )
        Q2 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand2, a=0, b=1000)[0]
        if error_boolean:
            error2 = 1 / np.pi * quad_vec(f=integrand2, a=0, b=1000)[1]

        price = s * Q1 - strike * np.exp(-self.r * time_to_maturity) * Q2
    
        if error_boolean:
            error = self.spot * error1 + strike * np.exp(-self.r * time_to_maturity) * error2
            return price, error
        else: 
            return price
        
    def call_price(
            self, 
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
            **kwargs
        ):
        """
        Price a European call option using the Carr-Madan method.

        This method prices a European call option using the Carr-Madan Fourier pricing method.

        :param np.array strike: Strike prices of the options.
        :param np.array time_to_maturity: Times to maturity of the options.
        :param np.array s: Current prices of the underlying asset.
        :param np.array v: Initial variances of the underlying asset.

        :returns: Call option prices.
        :rtype: float
        """
        
        price = self.carr_madan_price(
            s=s, 
            v=v,
            strike=strike, 
            time_to_maturity=time_to_maturity,
            **kwargs
        )
        return price
    
    
    def call_delta(
            self,
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
        ):
        """
        Calculate the delta of a European call option.

        This method calculates the delta of a European call option using the Heston model.

        :param np.array strike: Strike prices of the options.
        :param np.array time_to_maturity: Times to maturity of the options.
        :param np.array s: Current prices of the underlying asset.
        :param np.array v: Initial variances of the underlying asset.

        :returns: Delta of the call option.
        :rtype: float
        """
            
        if s is None:
            s = self.spot
        x = np.log(s)
        if v is None:
            v = self.vol_initial

        psi1 = self.characteristic(j=1)
        integrand = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi1(x, v, time_to_maturity, u)) / (u * 1j)
        )
        Q1 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand, a=0, b=1000)[0]

        return Q1

   
    def call_vega(
            self,
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
        ):
        """
        Calculate the vega of a European call option.

        This method calculates the vega of a European call option using the Heston model.

        :param np.array strike: Strike prices of the options.
        :param np.array time_to_maturity: Times to maturity of the options.
        :param np.array s: Current prices of the underlying asset.
        :param np.array v: Initial variances of the underlying asset.

        :returns: Vega of the call option.
        :rtype: float
        """

        if s is None:   
            s = self.spot
        x = np.log(s)
        if v is None:
            v = self.vol_initial

        u1 = 1 / 2
        b1 = self.kappa + self.drift_emm - self.rho * self.sigma
        u2 = -1 / 2
        b2 = self.kappa + self.drift_emm

        d1 = lambda u: np.sqrt(
            (self.rho * self.sigma * u * 1j - b1) ** 2
            - self.sigma**2 * (2 * u1 * u * 1j - u**2)
        )
        d2 = lambda u: np.sqrt(
            (self.rho * self.sigma * u * 1j - b2) ** 2
            - self.sigma**2 * (2 * u2 * u * 1j - u**2)
        )
        g1 = lambda u: (self.rho * self.sigma * u * 1j - b1 - d1(u)) / (
            self.rho * self.sigma * u * 1j - b1 + d1(u)
        )
        g2 = lambda u: (self.rho * self.sigma * u * 1j - b2 - d2(u)) / (
            self.rho * self.sigma * u * 1j - b2 + d2(u)
        )
        D1 = (
            lambda tau, u: (b1 - self.rho * self.sigma * u * 1j + d1(u))
            / self.sigma**2
            * (1 - np.exp(d1(u) * tau))
            / (1 - g1(u) * np.exp(d1(u) * tau))
        )
        D2 = (
            lambda tau, u: (b2 - self.rho * self.sigma * u * 1j + d2(u))
            / self.sigma**2
            * (1 - np.exp(d2(u) * tau))
            / (1 - g2(u) * np.exp(d2(u) * tau))
        )

        psi1 = self.characteristic(j=1)
        integrand1 = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi1(x, v, time_to_maturity, u) * D1(time_to_maturity, u)) / (u * 1j)
        )
        integral1 = 1 / np.pi * quad_vec(f=integrand1, a=0, b=1000)[0]

        psi2 = self.characteristic(j=2)
        integrand2 = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi2(x, v, time_to_maturity, u) * D2(time_to_maturity, u)) / (u * 1j)
        )
        integral2 = 1 / np.pi * quad_vec(f=integrand2, a=0, b=1000)[0]

        return s * integral1 - strike * np.exp(-self.r * time_to_maturity) * integral2

    
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

        :param np.array strike: Strike prices of the options.
        :param np.array time_to_maturity: Times to maturity of the options.
        :param np.array s: Current prices of the underlying asset.
        :param np.array v: Initial variances of the underlying asset.
        :param bool error_boolean: Whether to return the error of the price calculation.

        :returns: Option prices and errors (if error_boolean is True).
        :rtype: float or tuple
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
   
    def price_surface(self):
        """
        Plot the call price surface as a function of strike and time to maturity.

        This method generates a 3D plot of the call option prices for a range of strikes and maturities.

        :returns: None
        """

        Ks = np.linspace(start=20, stop=200, num=200)
        Ts = np.linspace(start=0.1, stop=2, num=200)
        K_mesh, T_mesh = np.meshgrid(Ks, Ts)

        call_prices = self.call_price(strike=K_mesh, time_to_maturity=T_mesh, s=self.spot, v=self.vol_initial)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(K_mesh, T_mesh, call_prices.T, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.set_title("Call price as a function of strike and time to maturity")
        ax.set_xlabel(r"Strike ($K$)")
        ax.set_ylabel(r"Time to maturity ($T$)")
        ax.set_zlabel("Price")
        ax.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.show()

    def delta_vega_hedging(
        self, 
        strike: float,
        strike_hedging: float,
        maturity: float,
        maturity_hedging: float,
        nbr_points: float = 252, 
        nbr_simulations: float = 100
    ):
        """
        Implement a delta-vega hedging strategy for a European option using the Heston model.

        This function simulates the hedging process over the lifetime of the option by dynamically rebalancing a portfolio
        consisting of a risky asset (underlying stock), an option (for vega hedging), and a non-risky asset (bank account).
        The function assumes that both the pricing and hedging models are based on the Heston stochastic volatility model,
        but they may use different volatilities for hedging and pricing.

        :param float strike: Strike price of the option to be hedged.
        :param float strike_hedging: Strike price of the option used for hedging.
        :param float maturity: Time to maturity of the option to be hedged.
        :param float maturity_hedging: Time to maturity of the option used for hedging.
        :param int nbr_points: Number of time points in the simulation.
        :param int nbr_simulations: Number of simulations to run.

        :returns: Portfolio values, underlying asset prices, variances, and option prices.
        :rtype: tuple
        """

        # Simulation
        time = np.linspace(start=0, stop=maturity, num=nbr_points + 1)
        time_to_maturities = np.tile(maturity - time, (nbr_simulations, 1))
        time_hedging = np.linspace(start=0, stop=maturity, num=nbr_points + 1)
        time_to_maturities_hedging = np.tile(maturity_hedging - time_hedging, (nbr_simulations, 1))
        dt = maturity / nbr_points
        r = self.r

        S, V, _ = self.simulate(
            time_to_maturity=maturity, 
            nbr_points=nbr_points, 
            nbr_simulations=nbr_simulations, 
            scheme='milstein'
        )
        portfolio = np.zeros_like(S)

        # Prices Calculation
        print("Computing option prices ...")
        C = self.call_price(strike=strike, time_to_maturity=time_to_maturities, s=S, v=V)
        C_hedging = self.call_price(strike=strike_hedging, time_to_maturity=time_to_maturities_hedging, s=S, v=V)

        # Vegas Calculation
        print("Computing vegas ...")
        vega = self.call_vega(strike=strike, time_to_maturity=time_to_maturities, s=S, v=V)
        vega_hedging = self.call_vega(strike=strike_hedging, time_to_maturity=time_to_maturities_hedging, s=S, v=V)

        # Deltas Calculation
        print("Computing deltas ...")
        delta = self.call_delta(strike=strike, time_to_maturity=time_to_maturities, s=S, v=V)
        delta_hedging = self.call_delta(strike=strike_hedging, time_to_maturity=time_to_maturities_hedging, s=S, v=V)
        
        # Delta-vega hedging
        stocks = np.zeros(nbr_simulations)
        derivatives = np.zeros(nbr_simulations)
        bank = np.zeros(nbr_simulations)

        portfolio[:, 0] = C[:, 0]
        derivatives = vega[:, 0] / vega_hedging[:, 0]
        stocks = delta[:, 0] - derivatives * delta_hedging[:, 0]
        bank = portfolio[:, 0] - stocks * S[:, 0] - derivatives * C_hedging[:, 0]

        for t in tqdm(range(1, nbr_points)):

            # Mise à jour de la banque
            bank = bank * np.exp(dt * r)

            # Mise à jour du portefeuille : valeur totale = banque + actions + dérivés
            portfolio[:, t] = bank + stocks * S[:, t] + derivatives * C_hedging[:, t]

            # Nouvelle couverture
            derivatives = vega[:, t] / vega_hedging[:, t]
            stocks = delta[:, t] - derivatives * delta_hedging[:, t]

            bank = portfolio[:, t] - stocks * S[:, t] - derivatives * C_hedging[:, t]

        portfolio[:, -1] = (
            bank * np.exp(dt * r) + stocks * S[:, -1] + derivatives * C_hedging[:, -1]
        )
        return portfolio, S, V, C