from hestonpy.models.utils import compute_smile
from hestonpy.models.blackScholes import BlackScholes
from hestonpy.models.svi import StochasticVolatilityInspired as SVI
fontdict = {'fontsize': 10, 'fontweight': 'bold'}

from scipy.optimize import minimize, basinhopping
from typing import Literal
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

class CustomStep:
    """
    Par défaut, basinhopping utilise un saut uniforme sur toutes les dimensions, ce qui peut être sous-optimal. 
    Un saut gaussien mieux peut être plus adapté à l’échelle de chaque paramètre :
    """

    def __init__(self, scale):
        self.scale = scale
    def __call__(self, x):
        return x + np.random.normal(scale=self.scale, size=len(x))  # Sauts gaussiens

class VolatilitySmile:
    """
    Represents a volatility smile constructed from market prices or implied volatilities.
    Handles the conversion between option prices and implied volatilities using the Black-Scholes model.
    Supports calibration of a Heston model to fit the observed volatility smile.
    """

    def __init__(
            self,
            strikes: np.array, 
            time_to_maturity: np.array,
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
            self.time_to_maturity = time_to_maturity

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
        return bs.call_price(strike=self.strikes, volatility=ivs, time_to_maturity=self.time_to_maturity)
    
    def compute_smile(self, prices:np.array = None, strikes:np.array = None):
        """
        Computes implied volatilities from option prices using the Black-Scholes model.
        
        Parameters:
        - prices (np.array, optional): option prices corresponding to the strikes.
        
        Returns:
        - np.array: Implied volatilities computed from the option prices.
        """
        if prices is None:
            prices = self.market_prices
        if strikes is None:
            strikes = self.strikes

        bs = BlackScholes(spot=self.atm, r=self.r, mu=self.r, volatility=0.02)
        smile = compute_smile(
            prices=prices, 
            strikes=strikes, 
            time_to_maturity=self.time_to_maturity,
            bs=bs,
            flag_option='call',
            method='dichotomie'
        )
        return smile
    
    def filters(
            self, 
            full_market_data: pd.DataFrame,
            select_mid_ivs: bool = True
        ):
        strikes = full_market_data['Strike'].values
        volumes = full_market_data['Volume'].values

        # Bid prices and implied vol
        bid_prices = full_market_data['Bid'].values
        bid_ivs = self.compute_smile(bid_prices, strikes)

        # Ask prices and implied vol
        ask_prices = full_market_data['Ask'].values
        ask_ivs = self.compute_smile(ask_prices, strikes)

        # Mid prices and implied vol
        mid_ivs = (ask_ivs + bid_ivs)/2
        mid_prices = (bid_prices + ask_prices)/2


        # 1st mask: trading volume more than 10 contracts
        mask1 = volumes >= 10

        # 2nd mask: mid-prices higher than 0.375
        mask2 = mid_prices >= 0.375

        # 3rd mask: bid-ask spread must be less than 10%
        spread = (ask_ivs - bid_ivs)/mid_ivs
        mask3 = (spread <= 0.30) & (mid_ivs < 0.9)

        # 4th mask: in- or out-of-the-money by more than 20% are excluded
        forward = self.atm * np.exp(self.r * self.time_to_maturity)
        mask4 = np.abs(self.strikes/forward - 1.0) <= 0.30

        masks = mask1 & mask2 & mask3 & mask4
        pd.options.mode.chained_assignment = None
        market_data = full_market_data.loc[masks]
        market_data['Mid ivs'] = mid_ivs[masks]
        market_data['Ask ivs'] = ask_ivs[masks]
        market_data['Bid ivs'] = bid_ivs[masks]
        market_data['Mid Price'] = mid_prices[masks]
        pd.options.mode.chained_assignment = 'warn'

        if select_mid_ivs:
            # Parameters
            self.strikes = market_data['Strike'].values
            self.market_ivs = market_data['Mid ivs'].values
            self.market_prices = market_data['Mid Price'].values

        return market_data
    
    def svi_smooth(self, select_svi_ivs: bool = False):
        """
        Smooth via a raw SVI
        """

        raw_svi = SVI(time_to_maturity=self.time_to_maturity)
        forward = self.atm * np.exp(self.time_to_maturity * self.r)
        calibrated_params, calibrated_ivs = raw_svi.calibration(
            strikes=self.strikes, market_ivs=self.market_ivs, forward=forward
        )
        if select_svi_ivs:
            self.market_ivs = calibrated_ivs
        return calibrated_params, calibrated_ivs
    
    def calibration(
            self,
            price_function,
            initial_guess: list = [1.25, 0.04, 0.25, 0.5],
            guess_correlation_sign: Literal['positive','negative','unknown'] = 'unknown',
            speed: Literal['local','global'] = 'local',
            weights: np.array = None,
            power: Literal['rmse', 'mae', 'mse'] = 'mse',
            relative_errors: bool = False,
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
        - price_function (callable): Function to compute option prices under the Heston model. Typically set as `price_function = heston.call_price`.
        - initial_guess (list): Initial parameters [kappa, theta, sigma, rho].
        - guess_correlation_sign (str): Assumption on the correlation sign ('positive', 'negative', 'unknown').
        - speed (str): Calibration method ('local' for fast, 'global' for robust optimization).
        - weights (np.array, optional): Array of weights applied to each observation in the calibration. 
          If None, uniform weights are used. Weights allow adjusting the importance of different strike prices in the calibration.
        - power (str, optional): Defines the loss function's exponentiation method. 
          Options:
          - 'mse': Uses squared differences (default).
          - 'mae': Uses absolute differences.
          - 'rmse': Uses square-root differences.
        - relative_errors (bool, optional): If True, the calibration minimizes relative errors instead of absolute errors.
          This can help normalize the impact of different strikes.
        - method (str): Optimization algorithm to use.

        Returns:
        - dict: Dictionary containing the calibrated Heston parameters.
        """
        if weights is None:
            nbr_observations = len(self.strikes)
            weights = np.array([1] * nbr_observations) / nbr_observations
        else:
            weights = weights / np.sum(weights)

        index_atm = np.argmin(np.abs(self.strikes - self.atm))
        vol_initial = self.market_ivs[index_atm]**2



        # Cost function and power, relative_errors parameters
        def difference_function(model_prices):
            if not relative_errors:
                difference = self.market_prices - model_prices
            else:
                difference = (self.market_prices - model_prices) / self.market_prices
            
            if power == 'mae':
                return np.sum(np.abs(difference))
            elif power == 'rmse':
                return np.sum(np.sqrt(difference**2))
            elif power == 'mse':
                return np.sum(difference**2)
            else: 
                raise ValueError("Invalid power. Choose either 'rmse', 'mae', or 'mse'.")

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
                    **function_params, v=vol_initial, strike=self.strikes, time_to_maturity=self.time_to_maturity, s=self.atm
                )
            
            return difference_function(model_prices)

        

        # Bounds of parameters
        bounds = [
            (1e-3, 10),   # kappa 
            (1e-3, 3), # theta
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
        


        
        # Fast/local calibration scheme
        if speed == 'local':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = minimize(cost_function, initial_guess, method=method, bounds=bounds)

        # Global calibration scheme
        elif speed == 'global':
            print(f"Initial Parameters: kappa={initial_guess[0]:.3f} | theta={initial_guess[1]:.3f} | sigma={initial_guess[2]:.3f} | rho={initial_guess[3]:.3f}\n")

            minimizer_kwargs = {
                "method": method,
                "bounds": bounds
            }
            def callback(x, f, accepted):
                if accepted:
                    print("at minimum %.6f accepted %d" % (f, accepted))
                    print(f"Parameters: v0={vol_initial:.3f} | kappa={x[0]:.3f} | theta={x[1]:.3f} | sigma={x[2]:.3f} | rho={x[3]:.3f}\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                step_taker = CustomStep(scale=[3/7, 1.5/7, 3/7, 1/7])
                result = basinhopping(cost_function, x0=initial_guess, callback=callback, minimizer_kwargs=minimizer_kwargs,
                                    niter=10, 
                                    niter_success=10,
                                    take_step=step_taker, 
                                    T=2.0)
                print(result.message, result.success)
        else: 
            raise ValueError("Invalid speed. Choose either 'local', or 'global'.")

        calibrated_params = {
            "vol_initial": vol_initial, 
            "kappa": result.x[0],
            "theta": result.x[1],
            "sigma": result.x[2],
            "rho": result.x[3],
            "drift_emm": 0,
        }
        
        print(f"Calibrated parameters: v0={vol_initial:.3f} | kappa={result.x[0]:.3f} | theta={result.x[1]:.3f} | sigma={result.x[2]:.3f} | rho={result.x[3]:.3f}\n")
        
        return calibrated_params
    
    def evaluate_calibration(self, model_values: np.array, metric_type: Literal["price", "iv"] = "price"):
        """
        Évalue la qualité de la calibration en calculant RMSE, MSE et MAE 
        soit sur les prix, soit sur les volatilités implicites (IVs).

        Parameters:
        - model_values (np.array): Valeurs estimées par le modèle (prix ou IVs).
        - metric_type (str): "price" pour comparer les prix, "iv" pour comparer les IVs.

        Returns:
        - dict: Contenant les métriques d'erreur absolues et relatives.
        """
        if metric_type == "price":
            actual_values = self.market_prices
        elif metric_type == "iv":
            actual_values = self.market_ivs * 100
            model_values = model_values * 100
        else:
            raise ValueError("metric_type must be either 'price' or 'iv'.")

        # Différences absolues et relatives
        diff = actual_values - model_values
        diff_rel = diff / (actual_values + 1e-8)  # Évite la division par zéro

        # Calcul des métriques
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(diff))

        mse_pct = np.mean(diff_rel ** 2) * 100
        rmse_pct = np.sqrt(mse_pct)
        mae_pct = np.mean(np.abs(diff_rel)) * 100

        return {
            "MSE": round(mse,3),
            "RMSE": round(rmse,3),
            "MAE": round(mae,3),
            "MSE_%": round(mse_pct,3),
            "RMSE_%": round(rmse_pct,3),
            "MAE_%": round(mae_pct,3)
        }

    
    def plot(
            self, 
            calibrated_ivs: np.array= None,
            calibrated_prices: np.array=None,
            bid_prices: np.array=None,
            bid_ivs: np.array=None,
            ask_prices: np.array=None,
            ask_ivs: np.array=None,
            maturity: str=None
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
        if (bid_ivs is None) and (bid_prices is not None):
            bid_ivs = self.compute_smile(prices=bid_prices)
        if (ask_ivs is None) and (ask_prices is not None):
            ask_ivs = self.compute_smile(prices=ask_prices)

        forward = self.atm * np.exp(self.r * self.time_to_maturity)

        plt.figure(figsize=(8, 5))

        plt.scatter(self.strikes/forward, self.market_ivs, label="data", marker='o', color='red', s=20)
        plt.axvline(1, linestyle="--", color="gray", label="ATM Strike")

        if calibrated_ivs is not None:
            plt.plot(self.strikes/forward, calibrated_ivs, label="calibred", marker='+', color='blue', linestyle="dotted", markersize=4)
        if bid_ivs is not None:
            plt.scatter(self.strikes/forward, bid_ivs, label="bid", marker=6, color='black', s=20)
        if ask_ivs is not None:
            plt.scatter(self.strikes/forward, ask_ivs, label="ask", marker=7, color='gray', s=20)

        plt.xlabel("Moneyness [%]", fontdict=fontdict)
        plt.ylabel("Implied Volatility [%]", fontdict=fontdict)

        if maturity is not None:
            date = datetime.strptime(maturity, '%Y-%m-%d').date().strftime("%d-%B-%y")
            title = f"{date}: {self.time_to_maturity * 252 / 21:.2f} mois"
        else:
            title = f"Time to maturity: {self.time_to_maturity * 252 / 21:.2f} mois"
        
        plt.title(title, fontdict=fontdict)
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.legend()
        plt.show()
 
