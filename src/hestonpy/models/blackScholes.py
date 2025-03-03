from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class BlackScholes:

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
        if spot is None:
            spot = self.spot
        if r is None:
            r = self.r
        if volatility is None:
            volatility = self.volatility

        call_price = self.call_price(spot, r, volatility, time_to_maturity, strike)
        put_price = call_price - spot + strike * np.exp(-r * time_to_maturity)
        return put_price

    def delta(
        self,
        strike: float,
        time_to_maturity: float,
        flag_option: Literal["call", "put"],
        spot: float = None,
        r: float = None,
        volatility: float = None,
    ):
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
        """
        Plot the delta of the option as a function of strike and time to maturity
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
        """ "
        Plot the gamma as a function of strike and time to maturity.
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
        """
        Implement a delta hedging strategy using both a risky asset (underlying asset)
        and a non-risky asset for a European option.
        Parameters:
            - flag_option (str):
                Type of option. Should be 'call' for a call option or 'put' for a put option.
            - strike (float):
                The strike price of the option.
            - hedging_volatility (float):
                The volatility used for hedging purposes.
            - nbr_hedges (float, optional):
                The number of simulation steps or trading intervals over the life
                of the option. Defaults to 1000. This parameter controls how often
                the portfolio is rebalanced to maintain a delta-neutral position.
            - nbr_simulations (float, optional):
                The number of simulations.
        Returns:
            - portfolio (np.array): allocation,
            - S (np.array):
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
        """
        Implement a volatility arbitrage strategy by buying an underpriced option
        and dynamically delta hedging it to expiry.

        Parameters:
            - flag_option (str):
                Type of option. Should be 'call' for a call option. Currently, only
                'call' options are supported.
            - T (float):
                The time to maturity of the option (in years).
            - strike (float):
                The strike price of the option.
            - hedging_volatility (float):
                The volatility used for delta hedging. This reflects the trader's
                belief about the actual market volatility.
            - pricing_volatility (float, optional):
                The implied volatility used to calculate the option price. If not
                provided, defaults to the value of `hedging_volatility`.
            - nbr_hedges (float, optional):
                The number of trading intervals over the life of the option. This
                parameter controls how frequently the portfolio is rebalanced to
                maintain the hedge. Defaults to 1000.
            - nbr_simulations (float, optional):
                The number of Monte Carlo simulations for the stock price path.
                Defaults to 100.

        Returns:
            - portfolio (np.array):
                An array representing the value of the portfolio at each step of
                the simulation. This includes the value of the option, stock
                holdings, and cash.
            - S (np.array):
                An array of simulated underlying asset prices over the life of the
                option.

        Explanation:
        This function models a scenario where the implied volatility of an option
        (used for pricing) differs from the trader's forecasted actual volatility
        (used for hedging). If the trader's forecast turns out to be correct, a
        profit can be realized by buying the option and dynamically delta hedging
        it. The delta for hedging is calculated using the trader's forecasted
        volatility, while the option price reflects the implied volatility.
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


# def dichotomie(price: np.array, bs:BlackScholes, error: float = 10**-6):

#     index_sup = 1
#     index_inf = 0 
#     volatilities = np.arange(start=0, stop=1, step=error)
#     index = len(volatilities) // 2
#     price_index = bs.call_price(volatilities[index])

#     while volatilities[index_sup] - volatilities[index_inf] > error:

#         price_index = bs.call_price(volatilities[index])
#         if price_index > price:

#             index_sup = index
#         else:
#             index_inf = index
#         index = index // 2

#     return volatilities[index]




# from scipy.optimize import minimize
# from hestonpy.option.data import get_options_data
# from typing import Literal


# def impliedVolatility(
#     flag_option: Literal["call", "put"],
#     symbol: str = "MSFT",
#     r: float = None,
#     exponent: int = 1,
# ):
#     """
#     Estimate the implied volatility of options using the Black-Scholes model.

#     Parameters:
#     - flag_option (Literal['call', 'put']): Type of option (call or put).
#     - symbol (str): The underlying asset's symbol; defaults to 'MSFT' (Microsoft Corporation).
#     - r (float): The risk-free interest rate; can be estimated if None.
#     - exponent (int): The exponent used in the objective function.
#                       If 2, it uses squared error; if 1, it uses absolute error.

#     Returns:
#     - vol: The estimated implied volatility (and possibly the risk-free rate if estimated).
#     """
#     # to do : implement for put options - pas super stable pour le moment

#     options_data, spot = get_options_data(symbol=symbol, flag_option=flag_option)

#     blackScholes = BlackScholes(spot=spot, r=0.03, mu=0.03, volatility=0.03)

#     mask = options_data["Volume"] > 0.1 * len(options_data)
#     options_data = options_data.loc[mask]

#     volumes = options_data["Volume"].values
#     strikes = options_data["Strike"].values
#     prices = options_data[f"{flag_option.capitalize()} Price"].values
#     maturities = options_data["Time to Maturity"].values

#     if r is None:
#         x0 = [blackScholes.volatility, blackScholes.r]
#     else:
#         # r is not estimated
#         x0 = [
#             blackScholes.volatility,
#         ]
#         blackScholes.r = r

#     def objective_function(x):
#         blackScholes.volatility = x[0]
#         if r is None:
#             # r is also estimated
#             blackScholes.r = x[1]

#         model_prices = []
#         for i in range(len(options_data)):
#             blackScholes.K = strikes[i]
#             blackScholes.T = maturities[i]

#             if flag_option == "call":
#                 model_price = blackScholes.call_price(
#                     strike=strikes[i], T=maturities[i]
#                 )
#             else:
#                 model_price = blackScholes.put_price(strike=strikes[i], T=maturities[i])
#             model_prices.append(model_price)

#         model_prices = np.array(model_prices)
#         weights = volumes / np.sum(volumes)

#         result = np.sum(weights * np.abs(prices - model_prices) ** exponent)

#         return result

#     print("Callibration is running...")
#     res = minimize(fun=objective_function, x0=x0, method="Nelder-Mead")

#     if res.success:
#         return res.x


# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Literal
# from scipy.optimize import minimize


# def volatility_surface(
#     flag_option: Literal["call", "put"],
#     symbol: str = "MSFT",
#     r: float = None,
#     exponent: int = 1,
# ):
#     """
#     Calculate the implied volatility surface from option prices.

#     Parameters:
#     - flag_option (Literal['call', 'put']): Type of option ('call' or 'put').
#     - symbol (str): The symbol of the underlying asset.
#     - r (float): Risk-free interest rate; can be None for estimation.
#     - exponent (int): Exponent used for optimization.

#     Returns:
#     - vol_surface: Matrix of implied volatilities.
#     """
#     # Not working very well

#     # Estimation of the interest rate
#     res = impliedVolatility(flag_option=flag_option, symbol=symbol, exponent=exponent)
#     r = res[1]

#     market_data, spot = get_options_data(symbol=symbol, flag_option=flag_option)
#     market_data = market_data.sort_values(
#         ["Strike", "Time to Maturity"], ascending=[True, True]
#     )

#     strikes = np.unique(market_data["Strike"])
#     maturities = np.unique(market_data["Time to Maturity"])

#     vol_surface = np.full((len(strikes), len(maturities)), np.nan)

#     def implied_volatility(market_price, strike, maturity):
#         """
#         Calcule la volatilité implicite pour une option donnée via une optimisation numérique.
#         """

#         def objective_function(vol):
#             bs = BlackScholes(spot=spot, r=r, mu=0, volatility=vol)
#             if flag_option == "call":
#                 model_price = bs.call_price(strike=strike, T=maturity)
#             else:
#                 model_price = bs.put_price(strike=strike, T=maturity)
#             return np.abs(model_price - market_price) ** exponent

#         result = minimize(objective_function, x0=0.2)
#         return result.x[0] if result.success else np.nan

#     for _, row in market_data.iterrows():
#         market_price = row[f"{flag_option.capitalize()} Price"]
#         strike = row["Strike"]
#         maturity = row["Time to Maturity"]

#         # Find indexes corresponding to (strike, maturity)
#         strike_index = np.where(strikes == strike)[0][0]
#         maturity_index = np.where(maturities == maturity)[0][0]

#         # Computed implied vol
#         iv = implied_volatility(market_price, strike, maturity)
#         vol_surface[strike_index, maturity_index] = iv

#     Strike_grid, Maturity_grid = np.meshgrid(strikes, maturities)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(
#         Strike_grid, Maturity_grid, vol_surface.T, cmap="viridis", edgecolor="none"
#     )
#     ax.set_xlabel("Strike")
#     ax.set_ylabel("Maturity")
#     ax.set_zlabel("Implied Volatility")
#     plt.title("Volatility Surface")
#     plt.show()

#     return vol_surface