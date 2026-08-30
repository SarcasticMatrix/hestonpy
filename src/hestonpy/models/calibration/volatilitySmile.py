from hestonpy.models.calibration._utils import compute_smile
from hestonpy.models.calibration._utils_optimisation import (
    _generate_difference_function,
    _get_parameters,
    _set_bounds,
    CustomStep,
    _feller,
    _get_calibrated_params,
    _callbacks
)
from hestonpy.models.blackScholes import BlackScholes
from hestonpy.models.calibration.svi import StochasticVolatilityInspired as SVI
from hestonpy.models.calibration.ssvi import SurfaceStochasticVolatilityInspired as SSVI

fontdict = {"fontsize": 20, "fontweight": "bold"}

from scipy.optimize import minimize, basinhopping, NonlinearConstraint
from dataclasses import dataclass
from typing import Literal
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


@dataclass
class ForwardDensity:
    """
    Forward (T-maturity) risk-neutral density of the underlying, recovered via the
    Breeden-Litzenberger formula f(K) = exp(r*T) * d^2C/dK^2, applied to a smoothed
    (SVI-fitted) call price curve evaluated on a fine, uniformly-spaced strike grid.

    Attributes:
        strikes (np.ndarray): Uniformly-spaced strike grid the density is evaluated on
            (the two grid endpoints are dropped, since the second derivative needs a
            neighbour on each side).
        density (np.ndarray): Risk-neutral density values at `strikes`.
        forward (float): Forward price used to build the strike grid (atm * exp(r*T)).
        time_to_maturity (float): Maturity in years.
        total_mass (float): Numerical integral of the density over `strikes` (trapezoidal
            rule). Should be close to 1; a value far from 1 signals a strike grid that's
            too narrow or too coarse relative to the smile's tails.
        has_negative_density (bool): True if some density values are negative, which
            signals butterfly arbitrage in the underlying smoothed price curve.
    """

    strikes: np.ndarray
    density: np.ndarray
    forward: float
    time_to_maturity: float
    total_mass: float
    has_negative_density: bool

    def plot(self):
        """
        Plots the forward risk-neutral density against strike. Points where the
        density is negative (butterfly arbitrage in the underlying smoothed price
        curve) are highlighted in red.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.strikes, self.density, color="blue", linewidth=1.5, label="Risk-neutral density")
        plt.fill_between(self.strikes, self.density, 0, color="blue", alpha=0.1)
        plt.axvline(self.forward, linestyle="--", color="gray", label="Forward")

        if self.has_negative_density:
            mask = self.density < 0
            plt.scatter(
                self.strikes[mask], self.density[mask],
                color="red", s=15, zorder=5, label="Negative density (arbitrage)",
            )

        plt.xlabel("Strike", fontdict=fontdict)
        plt.ylabel("Density", fontdict=fontdict)
        plt.title(
            f"Forward risk-neutral density (T={self.time_to_maturity:.2f}y)\nmass={self.total_mass:.3f}",
            fontdict=fontdict,
        )
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.legend()
        plt.show()


class VolatilitySmile:
    """
    Represents a volatility smile constructed from market prices or implied volatilities.

    This class handles the conversion between option prices and implied volatilities using
    the Black-Scholes model. It supports calibration of Heston and Bates models to fit the
    observed volatility smile.

    :param np.array strikes: Array of option strike prices.
    :param np.array time_to_maturity: Array of time to maturity for the options.
    :param float atm: At-the-money forward price.
    :param np.array market_prices: Array of market option prices.
    :param np.array market_ivs: Array of market implied volatilities.
    :param float r: Risk-free interest rate.
    """

    def __init__(
        self,
        strikes: np.array,
        time_to_maturity: np.array,
        atm: float,
        market_prices: np.array = None,
        market_ivs: np.array = None,
        r: float = 0,
    ):
        """
        Initializes the VolatilitySmile object.

        :param np.array strikes: Array of option strike prices.
        :param np.array time_to_maturity: Array of time to maturity for the options.
        :param float atm: At-the-money forward price.
        :param np.array market_prices: Array of market option prices.
        :param np.array market_ivs: Array of market implied volatilities.
        :param float r: Risk-free interest rate. Default is 0.
        """
        if market_prices is None and market_ivs is None:
            raise ValueError(
                "At least one of market_prices or market_ivs must be provided."
            )

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

    def reverse_smile(self, ivs: np.array = None) -> np.array:
        """
        Computes option prices from implied volatilities using the Black-Scholes model.

        :param np.array ivs: Implied volatilities corresponding to the strikes. If None, uses self.market_ivs.

        :returns: Option prices computed from the implied volatilities.
        :rtype: np.array
        """
        if ivs is None:
            ivs = self.market_ivs

        bs = BlackScholes(spot=self.atm, r=self.r, mu=self.r, volatility=0.02)
        return bs.call_price(
            strike=self.strikes, volatility=ivs, time_to_maturity=self.time_to_maturity
        )

    def compute_smile(self, prices: np.array = None, strikes: np.array = None) -> np.array:
        """
        Computes implied volatilities from option prices using the Black-Scholes model.

        :param np.array prices: Option prices corresponding to the strikes. If None, uses self.market_prices.
        :param np.array strikes: Strike prices. If None, uses self.strikes.

        :returns: Implied volatilities computed from the option prices.
        :rtype: np.array
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
            flag_option="call",
            method="dichotomie",
        )
        return smile

    def filters(self, full_market_data: pd.DataFrame, select_mid_ivs: bool = True) -> pd.DataFrame:
        """
        Filters market data based on volume, mid-price, bid-ask spread, and moneyness.

        :param pd.DataFrame full_market_data: DataFrame containing market data with columns: 'Strike', 'Volume', 'Bid', 'Ask'.
        :param bool select_mid_ivs: If True, selects mid implied volatilities. Default is True.

        :returns: Filtered market data with additional columns: 'Mid ivs', 'Ask ivs', 'Bid ivs', 'Mid Price'.
        :rtype: pd.DataFrame
        """
        strikes = full_market_data["Strike"].values
        volumes = full_market_data["Volume"].values

        # Bid prices and implied vol
        bid_prices = full_market_data["Bid"].values
        bid_ivs = self.compute_smile(bid_prices, strikes)

        # Ask prices and implied vol
        ask_prices = full_market_data["Ask"].values
        ask_ivs = self.compute_smile(ask_prices, strikes)

        # Mid prices and implied vol
        mid_ivs = (ask_ivs + bid_ivs) / 2
        mid_prices = (bid_prices + ask_prices) / 2

        # 1st mask: trading volume more than 10 contracts
        mask1 = volumes >= 10

        # 2nd mask: mid-prices higher than 0.375
        mask2 = mid_prices >= 0.375

        # 3rd mask: bid-ask spread must be less than
        spread = (ask_ivs - bid_ivs) / mid_ivs
        mask3 = (spread <= 0.30) & (ask_ivs < 0.9) & (mid_ivs < 0.5)

        # 4th mask: in- or out-of-the-money by more than 20% are excluded
        forward = self.atm * np.exp(self.r * self.time_to_maturity)
        mask4 = np.abs(self.strikes / forward - 1.0) <= 0.30

        masks = mask1 & mask2 & mask3 & mask4
        pd.options.mode.chained_assignment = None
        market_data = full_market_data.loc[masks]
        market_data["Mid ivs"] = mid_ivs[masks]
        market_data["Ask ivs"] = ask_ivs[masks]
        market_data["Bid ivs"] = bid_ivs[masks]
        market_data["Mid Price"] = mid_prices[masks]
        pd.options.mode.chained_assignment = "warn"

        if select_mid_ivs:
            # Parameters
            self.strikes = market_data["Strike"].values
            self.market_ivs = market_data["Mid ivs"].values
            self.market_prices = market_data["Mid Price"].values

        return market_data

    def svi_smooth(self, select_svi_ivs: bool = False):
        """
        Smooths the volatility smile using a raw SVI model.

        :param bool select_svi_ivs: If True, selects the calibrated SVI implied volatilities. Default is False.

        :returns: Dictionary of calibrated SVI parameters and array of calibrated implied volatilities.
        :rtype: Tuple[Dict[str, float], np.array]
        """

        raw_svi = SVI(time_to_maturity=self.time_to_maturity)
        forward = self.atm * np.exp(self.time_to_maturity * self.r)
        calibrated_params, calibrated_ivs = raw_svi.calibration(
            strikes=self.strikes, market_ivs=self.market_ivs, forward=forward
        )
        if select_svi_ivs:
            self.market_ivs = calibrated_ivs
        return calibrated_params, calibrated_ivs

    def forward_density(
        self,
        num_points: int = 500,
        strike_range: tuple[float, float] = None,
        svi_params: dict = None,
        ssvi_params: dict = None,
    ) -> ForwardDensity:
        """
        Computes the forward risk-neutral density for this maturity via the
        Breeden-Litzenberger formula: f(K) = exp(r*T) * d^2C/dK^2.

        Raw market strikes are too sparse and noisy for a stable second derivative,
        so the call price curve is instead built from a smooth fit of the smile,
        evaluated on a fine, uniformly-spaced strike grid (uniform spacing in
        strike, not log-moneyness, is required for the finite-difference second
        derivative to be accurate). Two parameterizations are supported:

        * ssvi_params={'rho', 'theta', 'phi'} (see ssvi.py): preferred when
          available, typically coming from VolatilitySurface.ssvi_smooth's joint
          fit — the wing function is then shared across the whole surface rather
          than fitted on this smile in isolation.
        * svi_params={'a', 'b', 'rho', 'm', 'sigma'} (raw SVI, see svi.py): used
          when ssvi_params is not given. If svi_params is also None, it is
          calibrated on the fly via svi_smooth() — this is the only option when a
          VolatilitySmile is used standalone, without a surrounding
          VolatilitySurface to provide a joint SSVI fit.

        :param int num_points: Number of points in the strike grid. Default is 500.
        :param tuple strike_range: (K_min, K_max) for the strike grid. If None,
            defaults to the market strike range padded by one span on each side,
            to capture enough of the tails for the integral to be close to 1.
        :param dict svi_params: Pre-calibrated raw SVI parameters (a, b, rho, m, sigma).
        :param dict ssvi_params: Pre-calibrated SSVI parameters (rho, theta, phi)
            for this maturity. Takes precedence over svi_params if both are given.

        :returns: The forward risk-neutral density for this maturity.
        :rtype: ForwardDensity
        """
        forward = self.atm * np.exp(self.r * self.time_to_maturity)

        if strike_range is None:
            span = self.strikes.max() - self.strikes.min()
            K_min = max(1e-6, self.strikes.min() - span)
            K_max = self.strikes.max() + span
        else:
            K_min, K_max = strike_range

        strikes_grid = np.linspace(K_min, K_max, num_points)
        log_moneyness = np.log(strikes_grid / forward)

        if ssvi_params is not None:
            ssvi = SSVI(maturities=np.array([self.time_to_maturity]))
            total_variance = ssvi.ssvi_total_variance(
                log_moneyness, ssvi_params["theta"], ssvi_params["rho"], ssvi_params["phi"]
            )
        else:
            if svi_params is None:
                svi_params, _ = self.svi_smooth()
            raw_svi = SVI(time_to_maturity=self.time_to_maturity)
            total_variance = raw_svi.raw_formulation(log_moneyness, **svi_params)

        ivs = np.sqrt(total_variance / self.time_to_maturity)

        bs = BlackScholes(spot=self.atm, r=self.r, mu=self.r, volatility=0.02)
        call_prices = bs.call_price(
            strike=strikes_grid, volatility=ivs, time_to_maturity=self.time_to_maturity
        )

        dK = strikes_grid[1] - strikes_grid[0]
        second_derivative = (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / dK**2
        density = np.exp(self.r * self.time_to_maturity) * second_derivative
        density_strikes = strikes_grid[1:-1]

        total_mass = np.trapezoid(density, density_strikes)

        return ForwardDensity(
            strikes=density_strikes,
            density=density,
            forward=forward,
            time_to_maturity=self.time_to_maturity,
            total_mass=total_mass,
            has_negative_density=bool(np.any(density < 0)),
        )

    def calibration(
        self,
        price_function,
        initial_guess,
        guess_correlation_sign: Literal["positive", "negative", "unknown"] = "unknown",
        speed: Literal["local", "global"] = "local",
        power: Literal["rmse", "mae", "mse"] = "mse",
        method: Literal["L-BFGS-B", "SLSQP", "trust-constr"] = "L-BFGS-B",
        weights: np.array = None,
        relative_errors: bool = False,
    ):
        """
        Calibrates a Heston model (parameters: kappa, theta, sigma, rho) or a Bates model (parameters: kappa, theta, sigma, rho, lambda_jump, mu_J, sigma_J) to fit the observed volatility smile.
        
        The initial variance is set to the closest ATM implied volatility from the data to reduce dimensionality.

        Two calibration schemes are available:

        * 'local': A fast but less robust method, sensitive to market noise.
        * 'global': A more robust but slower method.

        The user can specify a prior belief about the sign of the correlation:

        * 'positive': Constrains rho to [0,1].
        * 'negative': Constrains rho to [-1,0].
        * 'unknown': Allows rho to vary in [-1,1].
        
        If a correlation sign is provided, the function ensures the initial guess for rho has the correct sign.

        :param callable price_function: Function to compute option prices under the Heston model or Bates model.
        :param list initial_guess: Initial parameters, [kappa, theta, sigma, rho] for Heston models or [kappa, theta, sigma, rho, lambda_jump, mu_J, sigma_J] for Bates models.
        :param str guess_correlation_sign: Assumption on the correlation sign ('positive', 'negative', 'unknown').
        :param str speed: Calibration method ('local' for fast, 'global' for robust optimization).
        :param np.array weights: Array of weights applied to each observation in the calibration. If None, uniform weights are used.
        :param str power: Defines the loss function's exponentiation method ('mse', 'mae', 'rmse').
        :param bool relative_errors: If True, the calibration minimizes relative errors instead of absolute errors.
        :param str method: Local optimization algorithm to use ("L-BFGS-B", "SLSQP" or "trust-constr").

        :returns: Dictionary containing the calibrated Heston parameters.
        :rtype: dict
        """
        if weights is None:
            weights = 1 / len(self.strikes)
        else:
            weights = weights / np.sum(weights)

        if len(initial_guess) == 4:
            model_type = "Heston"
        elif len(initial_guess) == 7:
            model_type = "Bates"
        else:
            raise ValueError(
                "Invalid number of parameters in initial_guess. Must corresponds to 'Heston' or 'Bates'."
            )

        ########################################
        #### estimate v0
        ########################################
        index_atm = np.argmin(np.abs(self.strikes - self.atm))
        vol_initial = self.market_ivs[index_atm] ** 2

        ########################################
        #### set difference function, cost function, and optmisation parameters
        ########################################
        difference_function = _generate_difference_function(
            power=power, relative_errors=relative_errors, weights=weights
        )

        def cost_function(params):
            function_params = _get_parameters(model_type, params)
            model_prices = price_function(
                **function_params,
                v=vol_initial,
                strike=self.strikes,
                time_to_maturity=self.time_to_maturity,
                s=self.atm,
            )

            return difference_function(self.market_prices, model_prices)

        bounds = _set_bounds(model_type, guess_correlation_sign, initial_guess)
        minimizer_kwargs = {
            "method": method,
            "bounds": bounds,
            "constraints": NonlinearConstraint(_feller, 0, 100),
        }

        ########################################
        #### Fast/local calibration scheme
        ########################################
        if speed == "local":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = minimize(cost_function, initial_guess, **minimizer_kwargs)

        ########################################
        #### Global calibration scheme
        ########################################
        elif speed == "global":

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                result = basinhopping(
                    cost_function,
                    x0=initial_guess,
                    callback=_callbacks(model_type),
                    minimizer_kwargs=minimizer_kwargs,
                    niter=10,
                    niter_success=4,
                    # stepsize=0.3,
                    take_step=CustomStep(model_type),
                    T=2.0,
                )
                print(result.message, result.success)
        else:
            raise ValueError("Invalid speed. Choose either 'local', or 'global'.")

        calibrated_params = _get_calibrated_params(optmisation_result=result, vol_initial=vol_initial, model_type=model_type)
        return calibrated_params

    def evaluate_calibration(
        self, model_values: np.array, metric_type: Literal["price", "iv"] = "price"
    ):
        """
        Evaluates the quality of the calibration by calculating RMSE, MSE, and MAE
        either on prices or implied volatilities (IVs).

        :param np.array model_values: Values estimated by the model (prices or IVs).
        :param str metric_type: 'price' to compare prices, 'iv' to compare IVs. Default is 'price'.

        :returns: Dictionary containing the absolute and relative error metrics.
        :rtype: Dict[str, float]
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
        mse = np.mean(diff**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(diff))

        mse_pct = np.mean(diff_rel**2) * 100
        rmse_pct = np.sqrt(mse_pct)
        mae_pct = np.mean(np.abs(diff_rel)) * 100

        return {
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "MSE_%": round(mse_pct, 3),
            "RMSE_%": round(rmse_pct, 3),
            "MAE_%": round(mae_pct, 3),
        }

    def plot(
        self,
        calibrated_ivs: np.array = None,
        calibrated_prices: np.array = None,
        bid_prices: np.array = None,
        bid_ivs: np.array = None,
        ask_prices: np.array = None,
        ask_ivs: np.array = None,
        maturity: str = None,
    ):
        """
        Plots the volatility smile.

        This function can either:
        * Plot the smile using only the market data provided in the constructor.
        * Plot the smile with additional calibrated data (either calibrated implied volatilities or prices).

        If `calibrated_prices` is provided, the function computes the corresponding implied volatilities
        before plotting.

        :param np.array calibrated_ivs: Calibrated implied volatilities. If provided, they will be plotted.
        :param np.array calibrated_prices: Calibrated option prices. If provided, they will be converted to IVs before plotting.
        :param np.array bid_prices: Bid prices. If provided, they will be converted to IVs before plotting.
        :param np.array bid_ivs: Bid implied volatilities. If provided, they will be plotted.
        :param np.array ask_prices: Ask prices. If provided, they will be converted to IVs before plotting.
        :param np.array ask_ivs: Ask implied volatilities. If provided, they will be plotted.
        :param str maturity: Maturity date in the format 'YYYY-MM-DD'. If provided, it will be used in the plot title.
        """

        if (calibrated_ivs is None) and (calibrated_prices is not None):
            calibrated_ivs = self.compute_smile(prices=calibrated_prices)
        if (bid_ivs is None) and (bid_prices is not None):
            bid_ivs = self.compute_smile(prices=bid_prices)
        if (ask_ivs is None) and (ask_prices is not None):
            ask_ivs = self.compute_smile(prices=ask_prices)

        forward = self.atm * np.exp(self.r * self.time_to_maturity)

        plt.figure(figsize=(8, 5))

        plt.scatter(
            self.strikes / forward,
            self.market_ivs,
            label="data",
            marker="o",
            color="red",
            s=20,
        )
        plt.axvline(1, linestyle="--", color="gray", label="ATM Strike")

        if calibrated_ivs is not None:
            plt.plot(
                self.strikes / forward,
                calibrated_ivs,
                label="calibred",
                marker="+",
                color="blue",
                linestyle="dotted",
                markersize=4,
            )
        if bid_ivs is not None:
            plt.scatter(
                self.strikes / forward,
                bid_ivs,
                label="bid",
                marker=6,
                color="black",
                s=20,
            )
        if ask_ivs is not None:
            plt.scatter(
                self.strikes / forward,
                ask_ivs,
                label="ask",
                marker=7,
                color="gray",
                s=20,
            )

        plt.xlabel("Moneyness [%]", fontdict=fontdict)
        plt.ylabel("Implied Volatility [%]", fontdict=fontdict)

        if maturity is not None:
            date = datetime.strptime(maturity, "%Y-%m-%d").date().strftime("%d-%B-%y")
            title = f"{date}: {self.time_to_maturity * 252 / 21:.2f} mois"
        else:
            title = f"Time to maturity: {self.time_to_maturity * 252 / 21:.2f} mois"

        plt.title(title, fontdict=fontdict)
        plt.grid(
            visible=True,
            which="major",
            linestyle="--",
            dashes=(5, 10),
            color="gray",
            linewidth=0.5,
            alpha=0.8,
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":

    from hestonpy.models.heston import Heston

    # Synthetic market: call prices generated from a Heston model with known parameters
    true_params = dict(
        spot=100, vol_initial=0.04, r=0.02,
        kappa=2.0, theta=0.05, drift_emm=0.0, sigma=0.4, rho=-0.6,
    )
    heston = Heston(**true_params)

    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    time_to_maturity = 0.5
    market_prices = heston.call_price(strike=strikes, time_to_maturity=time_to_maturity)

    smile = VolatilitySmile(
        strikes=strikes,
        time_to_maturity=time_to_maturity,
        atm=100.0,
        market_prices=market_prices,
        r=0.02,
    )
    print(f"Market implied vols: {smile.market_ivs}\n")

    # Smooth the smile with a raw SVI fit
    svi_params, svi_ivs = smile.svi_smooth()
    print(f"SVI parameters: {svi_params}\n")
    smile.plot(calibrated_ivs=svi_ivs)

    # Recover the forward risk-neutral density via Breeden-Litzenberger
    density = smile.forward_density(svi_params=svi_params)
    print(f"Density total mass: {density.total_mass:.4f} | negative density: {density.has_negative_density}\n")
    density.plot()

    # Calibrate a Heston model back from the smile
    calibrated = smile.calibration(
        price_function=heston.call_price,
        initial_guess=[1.0, 0.03, 0.3, -0.3],
        guess_correlation_sign="negative",
        speed="local",
    )
    print(f"Calibrated Heston parameters: {calibrated}")
    print(f"True parameters: {true_params}")
