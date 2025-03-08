from hestonpy.models.utils import compute_smile
from hestonpy.models.blackScholes import BlackScholes
from scipy.optimize import minimize, basinhopping
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import warnings

class VolatilitySmile:
    """
    Represents a volatility smile constructed from market prices or implied volatilities.
    Handles the conversion between option prices and implied volatilities using the Black-Scholes model.
    Supports calibration of a Heston model to fit the observed volatility smile.
    """

    def __init__(
            self,
            strikes: np.array, 
            maturity: np.array,
            atm: float,
            market_prices: np.array = None,
            market_ivs: np.array = None,
            r: float = 0
        ):
            if market_prices is None and market_ivs is None:
                raise ValueError("At least one of market_prices or market_ivs must be provided.")
                  
            # Grid variables
            self.strikes = strikes 

            # Market variables: market_prices or market_ivs can be None
            self.market_prices = market_prices
            self.market_ivs = market_ivs
            self.atm = atm
            self.maturity = maturity

            # Model variables
            self.r = r

            if market_prices is None:
                self.market_prices = self.reverse_smile()
            if market_ivs is None:
                self.market_ivs = self.compute_smile()

    
    def reverse_smile(self, ivs:np.array = None):
        """
        Computes option prices from implied volatilities using the Black-Scholes model.
        
        Parameters:
        - ivs (np.array, optional): Implied volatilities corresponding to the strikes.
        
        Returns:
        - np.array: Option prices computed from the implied volatilities.
        """
        if ivs is None:
            ivs = self.market_ivs
        
        bs = BlackScholes(spot=self.atm, r=self.r, mu=self.r, volatility=0.02)
        return bs.call_price(strike=self.strikes, volatility=ivs, time_to_maturity=self.maturity)
    
    def compute_smile(self, prices:np.array=None):
        """
        Computes implied volatilities from option prices using the Black-Scholes model.
        
        Parameters:
        - prices (np.array, optional): Market option prices corresponding to the strikes.
        
        Returns:
        - np.array: Implied volatilities computed from the option prices.
        """
        if prices is None:
            prices = self.market_prices
        
        bs = BlackScholes(spot=self.atm, r=self.r, mu=self.r, volatility=0.02)
        smile = compute_smile(
            prices=prices, 
            strikes=self.strikes, 
            time_to_maturity=self.maturity,
            bs=bs,
            flag_option='call',
            method='dichotomie'
        )
        return smile
    
   
    def calibration(
            self,
            price_function,
            initial_guess: list = [1.25, 0.04, 0.25, 0.5],
            guess_correlation_sign: str = Literal['positive','negative','unknown'],
            speed: str = Literal['local','global'],
            method: str = 'L-BFGS-B'
        ):
        """
        Calibrates a Heston model (parameters: kappa, theta, sigma, rho) to fit the observed volatility smile.
        The initial variance is set to the closest ATM implied volatility from the data to reduce dimensionality.

        Two calibration schemes are available:
        - 'local': A fast but less robust method, sensitive to market noise.
        - 'global': A more robust but slower method.

        The user can specify a prior belief about the sign of the correlation:
        - 'positive': Constrains rho to [0,1].
        - 'negative': Constrains rho to [-1,0].
        - 'unknown': Allows rho to vary in [-1,1].

        If a correlation sign is provided, the function ensures the initial guess for rho has the correct sign.

        Parameters:
        - price_function (callable): Function to compute option prices under the Heston model. `price_function` is typically set as `price_function = heston.call_price`
        - initial_guess (list): Initial parameters [kappa, theta, sigma, rho].
        - guess_correlation_sign (str): Assumption on the correlation sign ('positive', 'negative', 'unknown').
        - speed (str): Calibration method ('local' for fast, 'global' for robust optimization).
        - method (str): Optimization algorithm to use.

        Returns:
        - dict: Calibrated Heston parameters.
        """


        index_atm = np.argmin(np.abs(self.strikes - self.atm))
        vol_initial = self.market_ivs[index_atm]**2

        def cost_function(params):
            kappa, theta, sigma, rho = params

            function_params = {
                "kappa": kappa,
                "theta": theta,
                "drift_emm": 0, 
                "sigma": sigma,
                "rho": rho,
            }
            
            model_prices = price_function(
                    **function_params, v=vol_initial, strike=self.strikes, time_to_maturity=self.maturity, s=self.atm
                )
            return np.sum((model_prices - self.market_prices) ** 2)
        
        # Bounds of parameters
        bounds = [
            (1e-3, 5),
            (1e-3, 2),
            (1e-3, 2),
        ]
        if guess_correlation_sign == 'positive':
            bounds.append((0.0,1.0))
            initial_guess[-1] = initial_guess[-1]
        elif guess_correlation_sign == 'negative':
            bounds.append((-1.0, 0.0))
            initial_guess[-1] = - initial_guess[-1]
        elif guess_correlation_sign == 'unknown':
            bounds.append((-1.0,1.0))

        # Fast calibration scheme
        if speed == 'local':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = minimize(cost_function, initial_guess, method=method, bounds=bounds)

        # Second calibration scheme
        elif speed == 'global':
            minimizer_kwargs = {
                "method": method,
                "bounds": bounds
            }
            def callback(x, f, accepted):
                print("at minimum %.6f accepted %d" % (f, accepted))
                print(f"Parameters: kappa={x[0]} | theta={x[1]} | sigma={x[2]} | rho={x[3]}\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = basinhopping(
                    cost_function, 
                    x0=initial_guess,
                    niter=10,
                    stepsize=0.3,
                    niter_success=4,
                    minimizer_kwargs=minimizer_kwargs,
                    callback=callback
                )

        calibrated_params = {
            "vol_initial": vol_initial, 
            "kappa": result.x[0],
            "theta": result.x[1],
            "drift_emm": 0,
            "sigma": result.x[2],
            "rho": result.x[3]
        }

        return calibrated_params
    
    
    def plot(
            self, 
            calibrated_ivs: np.array= None,
            calibrated_prices: np.array=None,
        ):
        """
        Plots the volatility smile.

        The function can either:
        - Plot the smile using only the market data provided in the constructor.
        - Plot the smile with additional calibrated data (either calibrated implied volatilities or prices).

        If `calibrated_prices` is provided, the function computes the corresponding implied volatilities 
        before plotting.

        Parameters:
        - calibrated_ivs (np.array, optional): Calibrated implied volatilities. If provided, they will be plotted.
        - calibrated_prices (np.array, optional): Calibrated option prices. If provided, they will be converted to IVs before plotting.
        """
        
        if (calibrated_ivs is None) and (calibrated_prices is not None):
            calibrated_ivs = self.compute_smile(prices=calibrated_prices)

        forward = self.atm * np.exp(self.r * self.maturity)

        plt.figure(figsize=(8, 5))

        plt.scatter(self.strikes/forward, self.market_ivs, label="syntetic data", marker='o', color='red', s=25)
        plt.axvline(1, linestyle="--", color="gray", label="ATM Strike")

        if calibrated_ivs is not None:
            plt.plot(self.strikes/forward, calibrated_ivs, label="calibred", marker='+', color='blue', markersize=4)

        plt.xlabel("Moneyness [%]")
        plt.ylabel("Implied Volatility [%]")
        plt.title("Volatility smile with syntetic data")
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.legend()
        plt.show()
 

def generate_smile(
        iv_atm: float = 0.2, 
        curvature: float = 0.0005, 
        skew: float = -0.002,
        strikes: np.array = None,
        atm: float = None,
        alea: bool = True
    ):
    """
    - sigma_atm: Volatilité implicite à la monnaie
    - curvature: Contrôle l’intensité du smile
    - skew: Contrôle l’inclinaison
    """
    
    if strikes is None:
        strikes = np.arange(start=80, stop=120, step=5)
    if atm is None:
        atm = 100

    market_ivs = iv_atm + skew * (strikes - atm) + curvature * (strikes - atm) ** 2
    if alea:
        market_ivs = (1 + np.random.normal(scale=0.01, size=len(market_ivs))) * market_ivs

    return market_ivs

if __name__ == "__main__":

    # Paramètres du marché synthétique
    atm = 100 
    r = 0.02
    strikes = np.arange(start=80, stop=120, step=5)
    iv_atm = 0.2
    curvature = 0.0005 
    skew = -0.002 

    markets_ivs = generate_smile(
        iv_atm, curvature, skew, strikes, atm
    )
     
    volSmile = VolatilitySmile(
        strikes=strikes,
        maturity=1,
        market_ivs=markets_ivs,
        atm=atm,
    )

    volSmile.plot()