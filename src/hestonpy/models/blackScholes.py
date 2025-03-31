from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class BlackScholes:
    """Black-Scholes model for option pricing.

    This class implements the Black-Scholes model for pricing European options.
    It includes methods for simulating asset paths, calculating option prices,
    and plotting Greeks.

    :param float spot: The current price of the underlying asset.
    :param float r: The risk-free interest rate.
    :param float mu: The expected return of the underlying asset.
    :param float volatility: The volatility of the underlying asset.
    :param int seed: Seed for the random number generator.
    """

    def __init__(
        self, spot: float, r: float, mu: float, volatility: float, seed: int = 42
    ):
        self.spot = spot
        self.r = r
        self.mu = mu
        self.volatility = volatility
        self.seed = seed

    def simulate(
        self,
        time_to_maturity: float = 1,
        scheme: str = Literal["euler", "milstein"],
        nbr_points: int = 100,
        nbr_simulations: int = 1000,
        ) -> np.array:
        """Simulate asset price paths using the Black-Scholes model.

        This method simulates the price paths of the underlying asset using
        either the Euler or Milstein discretization scheme.

        :param float time_to_maturity: Time to maturity of the option in years.
        :param str scheme: Discretization scheme to use ("euler" or "milstein").
        :param int nbr_points: Number of time points in each simulation.
        :param int nbr_simulations: Number of simulations to run.

        :returns: Simulated asset price paths.
        :rtype: np.array
        """

        np.random.seed(self.seed)

        dt = time_to_maturity / nbr_points
        S = np.zeros((nbr_simulations, nbr_points + 1))
        S[:, 0] = self.spot

        for i in range(1, nbr_points + 1):

            # Brownian motion
            Z = np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=nbr_simulations)

            # Update the processes
            S[:, i] = (
                S[:, i - 1]
                + self.mu * S[:, i - 1] * dt
                + self.volatility * S[:, i - 1] * Z
            )

            if scheme == "milstein":
                S[:, i] += 1 / 2 * self.volatility * S[:, i - 1] * (Z**2 - dt)

        if nbr_simulations == 1:
            S = S.flatten()

        return S

    def plot_simulation(
        self,
        scheme: str = Literal["euler", "milstein"],
        nbr_points: int = 1000,
        time_to_maturity: float = 1,
        ) -> np.array:
        """Plot a simulated asset price path.

        This method plots a single simulated price path of the underlying asset.

        :param str scheme: Discretization scheme to use ("euler" or "milstein").
        :param int nbr_points: Number of time points in the simulation.
        :param float time_to_maturity: Time to maturity of the option in years.

        :returns: Simulated asset price path.
        :rtype: np.array
        """
        
        S = self.simulate(nbr_points=nbr_points, scheme=scheme, nbr_simulations=1)

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, time_to_maturity, nbr_points + 1), S[0], label="Risky asset", color="blue", linewidth=1)
        plt.xlabel("Time to expiration", fontsize=12)
        plt.ylabel("Value [$]", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8,)
        plt.minorticks_on()
        plt.title(f"Black-Scholes Model Simulation with {scheme} scheme", fontsize=16)
        plt.show()

        return S

    def call_price(
        self,
        strike: float,
        time_to_maturity: float = 1,
        spot: float = None,
        r: float = None,
        volatility: float = None,
        ):
        """Calculate the price of a European call option.

        This method calculates the price of a European call option using the
        Black-Scholes formula.

        :param float strike: The strike price of the option.
        :param float time_to_maturity: Time to maturity of the option in years.
        :param float spot: The current price of the underlying asset.
        :param float r: The risk-free interest rate.
        :param float volatility: The volatility of the underlying asset.

        :returns: The price of the call option.
        :rtype: float
        """

        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility
        
        if time_to_maturity != 0: 
            d1 = (np.log(spot / strike) + (r + 0.5 * volatility**2) * time_to_maturity) / (
                volatility * np.sqrt(time_to_maturity)
            )
            d2 = d1 - volatility * np.sqrt(time_to_maturity)
            return spot * norm.cdf(d1) - strike * np.exp(-r * time_to_maturity) * norm.cdf(d2)
        else:
            return np.maximum(0, spot-strike)

    def put_price(
        self,
        strike: float,
        time_to_maturity: float,
        spot: float = None,
        r: float = None,
        volatility: float = None,
        ):
        """Calculate the price of a European put option.

        This method calculates the price of a European put option using the
        Black-Scholes formula.

        :param float strike: The strike price of the option.
        :param float time_to_maturity: Time to maturity of the option in years.
        :param float spot: The current price of the underlying asset.
        :param float r: The risk-free interest rate.
        :param float volatility: The volatility of the underlying asset.

        :returns: The price of the put option.
        :rtype: float
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility

        call_price = self.call_price(spot, r, volatility, time_to_maturity, strike)
        put_price = call_price - spot + strike * np.exp(-r * time_to_maturity)
        return put_price
    
    def vega(
        self,
        strike: float,
        time_to_maturity: float,
        spot: float = None,
        r: float = None,
        volatility: float = None,
        ):
        """Calculate the Vega of a European option.

        Vega measures the sensitivity of the option price to changes in volatility.

        :param float strike: The strike price of the option.
        :param float time_to_maturity: Time to maturity of the option in years.
        :param float spot: The current price of the underlying asset.
        :param float r: The risk-free interest rate.
        :param float volatility: The volatility of the underlying asset.

        :returns: The Vega of the option.
        :rtype: float
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility

        d1 = (np.log(spot / strike) + (r + 0.5 * volatility**2) * time_to_maturity) / (
            volatility * np.sqrt(time_to_maturity)
        )
        return spot * np.sqrt(time_to_maturity) * norm.pdf(d1)

    def delta(
        self,
        strike: float,
        time_to_maturity: float,
        flag_option: Literal["call", "put"],
        spot: float = None,
        r: float = None,
        volatility: float = None,
        ):
        """Calculate the Delta of a European option.

        Delta measures the sensitivity of the option price to changes in the
        underlying asset's price.

        :param float strike: The strike price of the option.
        :param float time_to_maturity: Time to maturity of the option in years.
        :param str flag_option: Type of option ("call" or "put").
        :param float spot: The current price of the underlying asset.
        :param float r: The risk-free interest rate.
        :param float volatility: The volatility of the underlying asset.

        :returns: The Delta of the option.
        :rtype: float
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility

        d1 = (
            np.log(spot / strike) + (r + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity)
        )

        if flag_option == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1


    def delta_surface(self, flag_option: Literal["call", "put"]):
        """Plot the Delta surface of the option as a function of strike and time to maturity.

        This method generates a 3D plot of the Delta for a range of strikes and
        times to maturity.

        :param str flag_option: Type of option ("call" or "put").
        """

        Ks = np.arange(start=20, stop=200, step=0.5)
        Ts = np.linspace(start=0.01, stop=1, num=500)
        deltas = np.zeros((len(Ks), len(Ts)))

        for i, K in enumerate(Ks):
            for j, T in enumerate(Ts):
                deltas[i, j] = self.delta(strike=K, time_to_maturity=T, flag_option=flag_option)
        K_grid, T_grid = np.meshgrid(Ks, Ts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(K_grid, T_grid, deltas.T, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Time to Maturity")
        ax.set_zlabel("Delta")
        plt.title("Delta Surface for European options")
        plt.show()

    def gamma(
        self,
        strike: float,
        time_to_maturity: float,
        spot: float = None,
        r: float = None,
        volatility: float = None,
        ):
        """Calculate the Gamma of a European option.

        Gamma measures the rate of change of Delta with respect to the underlying
        asset's price.

        :param float strike: The strike price of the option.
        :param float time_to_maturity: Time to maturity of the option in years.
        :param float spot: The current price of the underlying asset.
        :param float r: The risk-free interest rate.
        :param float volatility: The volatility of the underlying asset.

        :returns: The Gamma of the option.
        :rtype: float
        """
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility

        d1 = (np.log(spot / strike) + (r + 0.5 * volatility**2) * time_to_maturity) / (
            volatility * np.sqrt(time_to_maturity)
        )
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_maturity))
        return gamma

    def gamma_surface(self):
        """Plot the Gamma surface of the option as a function of strike and time to maturity.

        This method generates a 3D plot of the Gamma for a range of strikes and times to maturity.
        """

        Ks = np.arange(start=20, stop=200, step=0.5)
        Ts = np.linspace(start=0.01, stop=1, num=500)
        gammas = np.zeros((len(Ks), len(Ts)))

        for i, K in enumerate(Ks):
            for j, T in enumerate(Ts):
                gammas[i, j] = self.gamma(strike=K, time_to_maturity=T)
        K_grid, T_grid = np.meshgrid(Ks, Ts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(K_grid, T_grid, gammas.T, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Time to Maturity")
        ax.set_zlabel("Gamma")
        plt.title("Gamma Surface for European options")
        plt.show()

    def delta_hedging(
        self,
        strike: float,
        time_to_maturity: float,
        flag_option: Literal["call", "put"],
        hedging_volatility: float,
        pricing_volatility: float = None,
        nbr_hedges: float = 252,
        nbr_simulations: float = 100,
        ):
        """Implement a delta hedging strategy for a European option.

        This method simulates a delta hedging strategy using both a risky asset
        (underlying asset) and a non-risky asset.

        :param str flag_option: Type of option ("call" or "put").
        :param float strike: The strike price of the option.
        :param float hedging_volatility: The volatility used for hedging purposes.
        :param float pricing_volatility: The volatility used for pricing the option.
        :param int nbr_hedges: The number of hedging intervals.
        :param int nbr_simulations: The number of simulations.

        :returns: A tuple containing the portfolio value and simulated asset prices.
        :rtype: tuple
        """
        if pricing_volatility is None:
            pricing_volatility = hedging_volatility

        time = np.linspace(start=0, stop=time_to_maturity, num=nbr_hedges + 1)
        dt = time_to_maturity / nbr_hedges

        S = self.simulate(scheme="milstein", nbr_points=nbr_hedges, nbr_simulations=nbr_simulations)
        portfolio = np.zeros_like(S)

        if flag_option == "call":
            portfolio[:, 0] = self.call_price(
                strike=strike, time_to_maturity=time_to_maturity, spot=S[:, 0], volatility=pricing_volatility
            )
        else:
            portfolio[:, 0] = self.put_price(
                strike=strike, time_to_maturity=time_to_maturity, spot=S[:, 0], volatility=pricing_volatility
            )

        stocks = self.delta(
            spot=S[:, 0],
            time_to_maturity=time_to_maturity,
            volatility=hedging_volatility,
            strike=strike,
            flag_option=flag_option,
        )
        bank = portfolio[:, 0] - stocks * S[:, 0]

        for t in range(1, nbr_hedges):

            bank = bank * np.exp(dt * self.r)
            portfolio[:, t] = stocks * S[:, t] + bank

            stocks = self.delta(
                spot=S[:, t],
                time_to_maturity=time_to_maturity - time[t],
                volatility=hedging_volatility,
                strike=strike,
                flag_option=flag_option,
            )

            bank = portfolio[:, t] - stocks * S[:, t]

        portfolio[:, -1] = stocks * S[:, -1] + bank * np.exp(dt * self.r)
        return portfolio, S

    def volatility_arbitrage(
        self,
        strike: float,
        time_to_maturity: float,
        flag_option: Literal["call"],
        hedging_volatility: float,
        pricing_volatility: float = None,
        nbr_hedges: float = 1000,
        nbr_simulations: float = 100,
    ):
        """Implement a volatility arbitrage strategy by buying an underpriced option
        and dynamically delta hedging it to expiry.

        This function models a scenario where the implied volatility of an option 
        (used for pricing) differs from the trader’s forecasted actual volatility (used for hedging). 
        If the trader’s forecast turns out to be correct, a profit can be realized by buying the option 
        and dynamically delta hedging it. The delta for hedging is calculated using the trader’s forecasted 
        volatility, while the option price reflects the implied volatility.

        :param str flag_option: Type of option ("call").
        :param float strike: The strike price of the option.
        :param float hedging_volatility: The volatility used for delta hedging.
        :param float pricing_volatility: The implied volatility used for pricing.
        :param int nbr_hedges: The number of hedging intervals.
        :param int nbr_simulations: The number of simulations.

        :returns: A tuple containing the portfolio value and simulated asset prices.
        :rtype: tuple
        """

        if pricing_volatility is None:
            pricing_volatility = hedging_volatility

        time = np.linspace(start=0, stop=time_to_maturity, num=nbr_hedges + 1)
        dt = time_to_maturity / nbr_hedges

        S = self.simulate(scheme="milstein", nbr_points=nbr_hedges, nbr_simulations=nbr_simulations)
        portfolio = np.zeros_like(S)
        portfolio[:, 0] = 0  # Arbitrage

        C = lambda t, spot: self.call_price(
            volatility=pricing_volatility, spot=spot, time_to_maturity=time_to_maturity - t, strike=strike
        )
        delta = lambda t, spot: self.delta(
            volatility=hedging_volatility,
            spot=spot,
            time_to_maturity=time_to_maturity - t,
            flag_option="call",
            strike=strike,
        )

        stocks = delta(0, S[:, 0])
        bank = stocks * S[:, 0] - C(0, S[:, 0])

        for t in range(1, nbr_hedges):

            bank = bank * np.exp(dt * self.r)

            portfolio[:, t] = C(time[t], S[:, t]) - stocks * S[:, t] + bank

            stocks = delta(time[t], S[:, t])

            bank = portfolio[:, t] + stocks * S[:, t] - C(time[t], S[:, t])

        portfolio[:, -1] = (
            np.maximum(S[:, -1] - strike, 0)
            - stocks * S[:, -1]
            + bank * np.exp(dt * self.r)
        )

        return portfolio, S