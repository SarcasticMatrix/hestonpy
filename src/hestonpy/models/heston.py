import numpy as np
from numpy import random
from scipy.integrate import quad, quad_vec
from tqdm import tqdm
from typing import Literal

import matplotlib.pyplot as plt
from collections import namedtuple


class Heston:

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
        premium_volatility_risk=0.0,
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

        # World parameters
        self.r = r  # interest rate
        self.premium_volatility_risk = premium_volatility_risk

        self.seed = seed  # random seed

    def simulate(
        self,
        time_to_maturity: float = 1,
        scheme: str = Literal["euler", "milstein"],
        nbr_points: int = 100,
        nbr_simulations: int = 1000,
    ) -> tuple:

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
                + (self.r + self.premium_volatility_risk * np.sqrt(V[:, i - 1])) * S[:, i - 1] * dt
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
        ax1.minorticks_on()
        ax1.grid(which="minor", visible=False)

        ax2.plot(np.linspace(0, 1, nbr_points + 1),np.sqrt(V),label="Volatility",color="orange",linewidth=1,)
        ax2.axhline(y=np.sqrt(self.theta),label=r"$\sqrt{\theta}$",linestyle="--",color="black",)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Instantaneous volatility [%]", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8,)
        ax2.minorticks_on()
        ax2.grid(which="minor", visible=False)

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
    ) -> float:
        
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

    def characteristic(self, j: int) -> float:
        """
        Creates the characteristic function Psi_j(x, v, t; u) for a given (x, v, t).

        This function returns the characteristic function based on the index provided.

        Parameters:
        - j (int): Index of the characteristic function (must be 1 or 2).

        Returns:
        - callable: The characteristic function.
        """

        if j == 1:
            uj = 1 / 2
            bj = self.kappa + self.drift_emm - self.rho * self.sigma
        elif j == 2:
            uj = -1 / 2
            bj = self.kappa + self.drift_emm
        else:
            print("Argument j (int) must be 1 or 2")
            return 0
        a = self.kappa * self.theta

        dj = lambda u: np.sqrt(
            (self.rho * self.sigma * u * 1j - bj) ** 2
            - self.sigma**2 * (2 * uj * u * 1j - u**2)
        )
        gj = lambda u: (self.rho * self.sigma * u * 1j - bj - dj(u)) / (
            self.rho * self.sigma * u * 1j - bj + dj(u)
        )

        Cj = lambda tau, u: self.r * u * tau * 1j + a / self.sigma**2 * (
            (bj - self.rho * self.sigma * u * 1j + dj(u)) * tau
            - 2 * np.log((1 - gj(u) * np.exp(dj(u) * tau)) / (1 - gj(u)))
        )
        Dj = (
            lambda tau, u: (bj - self.rho * self.sigma * u * 1j + dj(u))
            / self.sigma**2
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
            error_boolean: bool = False
        ):
        if s is None:
            s = self.spot
        x = np.log(s)
        if v is None:
            v = self.vol_initial

        psi1 = self.characteristic(j=1)
        integrand1 = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi1(x, v, time_to_maturity, u)) / (u * 1j)
        )
        Q1 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand1, a=0, b=100)[0]
        if error_boolean:
            error1 = 1 / np.pi * quad_vec(f=integrand1, a=0, b=100)[1]

        psi2 = self.characteristic(j=2)
        integrand2 = lambda u: np.real(
            (np.exp(-u * np.log(strike) * 1j) * psi2(x, v, time_to_maturity, u)) / (u * 1j)
        )
        Q2 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand2, a=0, b=100)[0]
        if error_boolean:
            error2 = 1 / np.pi * quad_vec(f=integrand2, a=0, b=100)[1]

        price = self.spot * Q1 - strike * np.exp(-self.r * time_to_maturity) * Q2
    
        if error_boolean:
            error = self.spot * error1 + strike * np.exp(-self.r * time_to_maturity) * error2
            return price, error
        else: 
            return price
        
    def call_price(self):
        
        price_function = lambda strike, time_to_maturity, s, v: self.fourier_transform_price(
            s=s, 
            v=v,
            strike=strike, 
            time_to_maturity=time_to_maturity
        )
        return price_function
    
    
    def call_delta(self):

        def delta(
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
        ):
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

        delta_function = lambda strike, time_to_maturity, s, v: delta(
            s=s, 
            v=v,
            strike=strike, 
            time_to_maturity=time_to_maturity
        )

        price_function = self.call_price()
        delta_function = lambda strike, time_to_maturity, s, v: (
            price_function(s=s*1.01, v=v,strike=strike, time_to_maturity=time_to_maturity) - price_function(s=s*0.99, v=v,strike=strike, time_to_maturity=time_to_maturity)) / (s * 2/100)
        return delta_function

   
    def call_vega(self):

        def vega(
                strike: np.array, 
                time_to_maturity: np.array,
                s: np.array = None,
                v: np.array = None,
            ):
            if s is None:   
                s = self.spot
            x = np.log(s)
            if v is None:
                v = self.vol_initial

            psi1 = self.characteristic(j=1)
            integrand1 = lambda u: np.real(
                (np.exp(-u * np.log(strike) * 1j) * psi1(x, v, time_to_maturity, u)) / (u * 1j)
            )
            Q1 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand1, a=0, b=1000)[0]

            psi2 = self.characteristic(j=2)
            integrand2 = lambda u: np.real(
                (np.exp(-u * np.log(strike) * 1j) * psi2(x, v, time_to_maturity, u)) / (u * 1j)
            )
            Q2 = 1 / 2 + 1 / np.pi * quad_vec(f=integrand2, a=0, b=1000)[0]

            return s * Q1 - strike * np.exp(-self.r * time_to_maturity) * Q2

        vega_function = lambda strike, time_to_maturity, s, v: vega(
            s=s, 
            v=v,
            strike=strike, 
            time_to_maturity=time_to_maturity
        )
        price_function = self.call_price()
        vega_function = lambda strike, time_to_maturity, s, v: (
            price_function(s=s, v=v+(1/100)**2+2*1/100*v,strike=strike, time_to_maturity=time_to_maturity) - price_function(s=s, v=v,strike=strike, time_to_maturity=time_to_maturity))
        return vega_function
    

    def carr_madan_price(self, strike:float, time_to_maturity:float, error_boolean: bool = False):
        """
        Computes the price of a European call option using the Carr-Madan Fourier pricing method.

        This method employs the Carr-Madan approach, leveraging the characteristic function to calculate
        the option price.

        Returns:
        - price (float): The calculated option price.
        - error (float): The error associated with the option price calculation.
        """

        x = np.log(self.spot)
        v = self.vol_initial
        alpha = 0.3

        price_hat = (
            lambda u: np.exp(-self.r * time_to_maturity)
            / (alpha**2 + alpha - u**2 + u * (2 * alpha + 1) * 1j)
            * self.characteristic(j=2)(x, v, time_to_maturity, u - (alpha + 1) * 1j)
        )

        integrand = lambda u: np.exp(-u * np.log(strike) * 1j) * price_hat(u)

        price = (
            np.exp(-alpha * np.log(strike)) / np.pi * quad(func=integrand, a=0, b=50)[0]
        )

        if error_boolean:
            error = (
                np.exp(-alpha * np.log(strike)) / np.pi * quad(func=integrand, a=0, b=50)[1]
            )
            return price, error
        else: 
            return price   
   
    def price_surface(self):

        Ks = np.linspace(start=20, stop=200, num=200)
        Ts = np.linspace(start=0.1, stop=2, num=200)
        K_mesh, T_mesh = np.meshgrid(Ks, Ts)

        full_call_price = self.call_price()
        call_prices = lambda strike, time_to_maturity: full_call_price(strike=strike, time_to_maturity=time_to_maturity, s=self.spot, v=self.vol_initial)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(K_mesh, T_mesh, call_prices(K_mesh, T_mesh).T, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.set_title("Call price as a function of strike and time to maturity")
        ax.set_xlabel(r"Strike ($K$)")
        ax.set_ylabel(r"Time to maturity ($T$)")
        ax.set_zlabel("Price")
        ax.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.show()

def delta_vega_hedging(
    heston: Heston, 
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

    Parameters:

    Returns:

    Methodology:
        1. Simulation
           - Simulates paths for the underlying asset prices (`S`) and volatilities (`V`) using the Heston model.
        2. Greeks Calculation
           - Calculates vega and delta for both the pricing and hedging models at each time step.
        3. Hedging
           - Implements a delta-vega hedging strategy:
               * `stocks` holds the number of units of the underlying asset.
               * `derivatives` holds the number of options used for vega hedging.
               * `bank` holds the amount of cash in a non-risky asset.
        4. Rebalancing
           - Rebalances the portfolio at each hedging interval to maintain the desired delta and vega neutrality.
        ```
    """

    # Simulation
    time = np.linspace(start=0, stop=maturity, num=nbr_points + 1)
    time_to_maturities = np.tile(maturity - time, (nbr_simulations, 1))
    time_hedging = np.linspace(start=0, stop=maturity, num=nbr_points + 1)
    time_to_maturities_hedging = np.tile(maturity_hedging - time_hedging, (nbr_simulations, 1))
    dt = maturity / nbr_points
    r = heston.r

    S, V, _ = heston.simulate(
        time_to_maturity=maturity, 
        nbr_points=nbr_points, 
        nbr_simulations=nbr_simulations, 
        scheme='milstein'
    )
    portfolio = np.zeros_like(S)

    # Prices Calculation
    print("Computing option prices ...")
    full_call_price = heston.call_price()
    call_price = lambda strike, time_to_maturity, s, v: full_call_price(strike=strike, time_to_maturity=time_to_maturity, s=s, v=v)
    C = call_price(strike, time_to_maturities, S, V)
    C_hedging = call_price(strike_hedging, time_to_maturities_hedging, S, V)
    print(C.shape)

    # Vegas Calculation
    print("Computing vegas ...")
    full_call_vega = heston.call_vega()
    call_vega = lambda strike, time_to_maturity, s, v: full_call_vega(strike=strike, time_to_maturity=time_to_maturity, s=s, v=v)
    vega = call_vega(strike, time_to_maturities, S, V)
    vega_hedging = call_vega(strike_hedging, time_to_maturities_hedging, S, V)

    # Deltas Calculation
    print("Computing deltas ...")
    full_call_delta = heston.call_delta()
    call_delta = lambda strike, time_to_maturity, s, v: full_call_delta(strike=strike, time_to_maturity=time_to_maturity, s=s, v=v)
    delta = call_delta(strike, time_to_maturities, S, V)
    delta_hedging = call_delta(strike_hedging, time_to_maturities_hedging, S, V)
    
    # Delta-vega hedging
    stocks = np.zeros(nbr_simulations)
    derivatives = np.zeros(nbr_simulations)
    bank = np.zeros(nbr_simulations)

    # Hedging and Rebalancing
    portfolio[:, 0] = C[:, 0]

    derivatives = vega[:, 0] / vega_hedging[:, 0]
    stocks = delta[:, 0] - derivatives * delta_hedging[:, 0]
    bank = portfolio[:, 0] - stocks * S[:, 0] - derivatives * C_hedging[:, 0]

    for t in tqdm(range(1, nbr_points)):

        # Mise à jour de la banque
        bank = bank * np.exp(dt * r)

        # Mise à jour du portefeuille : valeur totale = banque + actions + dérivés
        portfolio[:, t] = bank + stocks * S[:, t] + derivatives * C_hedging[:, t]

        # Calcul de la nouvelle couverture delta
        derivatives = vega[:, t] / vega_hedging[:, t]
        stocks = delta[:, t] - derivatives * delta_hedging[:, t]

        bank = portfolio[:, t] - stocks * S[:, t] - derivatives * C_hedging[:, t]

    portfolio[:, -1] = (
        bank * np.exp(dt * r) + stocks * S[:, -1] + derivatives * C_hedging[:, -1]
    )
    return portfolio, S, V, C


from datetime import datetime
from scipy.optimize import minimize
from hestonpy.option.data import get_options_data


def calibrate(
    flag_option: str,
    heston: Heston,
    symbol: str = "MSFT",
):
    """
    Calibrates the Heston model using options data for various expiration dates and associated strikes.

    Parameters:
    - flag_option (str): Specifies the type of option (e.g., call or put).
    - heston (Heston): An instance of the Heston model to calibrate.
    - symbol (str): The stock symbol for which to gather options data; defaults to 'MSFT' for Microsoft Corporation.

    Returns:
    - res: The result of the optimization process, containing the optimized parameters for the Heston model.
    """
    # to do : implement for put options

    start_date = datetime.now()

    options_data, spot = get_options_data(symbol=symbol, flag_option=flag_option)
    heston.spot = spot

    # TEST
    mask = options_data["Volume"] > 0.1 * len(options_data)
    options_data = options_data.loc[mask]

    volumes = options_data["Volume"].values
    strikes = options_data["Strike"].values
    prices = options_data["Call Price"].values
    maturities = options_data["Time to Maturity"].values

    x0 = [
        heston.kappa,
        heston.theta,
        heston.sigma,
        heston.rho,
        heston.drift_emm,
        heston.vol_initial,
    ]

    def objective_function(x):
        heston.kappa = x[0]
        heston.theta = x[1]
        heston.sigma = x[2]
        heston.rho = x[3]
        heston.drift_emm = x[4]
        heston.vol_initial = x[5]

        model_prices = []
        for i in range(len(options_data)):
            heston.K = strikes[i]
            heston.T = maturities[i]
            model_price, _ = heston.fourier_transform_price()
            model_prices.append(model_price)

        model_prices = np.array(model_prices)
        weights = volumes / np.sum(volumes)

        result = np.sum(weights * (prices - model_prices) ** 2)

        return result

    print("Callibration is running...")
    res = minimize(fun=objective_function, x0=x0, method="Nelder-Mead")

    return res