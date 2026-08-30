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
from hestonpy.models.calibration.volatilitySmile import VolatilitySmile, ForwardDensity

fontdict = {"fontsize": 20, "fontweight": "bold"}

from scipy.optimize import minimize, basinhopping, NonlinearConstraint
from scipy.interpolate import interp1d, RectBivariateSpline, PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Literal
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


class VolatilitySurface:
    """
    Represents a volatility surface constructed from market prices or implied volatilities.

    A surface is simply a collection of VolatilitySmile objects, one per maturity.
    Smoothing (svi_smooth) is done independently for each maturity, reusing
    VolatilitySmile's own SVI fit. Model calibration (calibration) instead fits a
    single set of Heston/Bates parameters shared across every maturity at once,
    since kappa/theta/sigma/rho (and v0) are properties of the underlying dynamics,
    not of a single expiry.

    TODO: svi_smooth still calibrates each slice independently, with no constraint
    tying maturities together, so nothing enforces the absence of calendar-spread
    arbitrage across the surface (total implied variance w(k,T) = iv(k,T)**2 * T
    should be non-decreasing in T at fixed log-moneyness k) when using it. Either
    add a post-hoc check across consecutive maturities, or prefer ssvi_smooth, which
    now fits a single rho and wing function jointly across every maturity and
    enforces a sufficient no-arbitrage condition (Gatheral & Jacquier, 2014) by
    construction — see ssvi.py's calibrate_joint.
    """

    def __init__(
        self,
        smiles: list[VolatilitySmile]
    ):
        """
        Initializes the VolatilitySurface object as a list of smiles.
        """

        self.smiles = smiles
        self.time_to_maturities = [smile.time_to_maturity for smile in self.smiles]

    def compute_surface(self) -> pd.DataFrame:
        """
        Computes implied volatilities for every smile in the surface and stacks them
        into a DataFrame indexed by time to maturity, one column per observed strike.
        Strikes not quoted for a given maturity are left as NaN.

        :returns: Implied volatility surface (rows: maturities, columns: strikes).
        :rtype: pd.DataFrame
        """

        all_strikes = sorted({strike for smile in self.smiles for strike in smile.strikes})
        surface = pd.DataFrame(index=self.time_to_maturities, columns=all_strikes, dtype=float)

        for smile, ttm in zip(self.smiles, self.time_to_maturities):
            ivs = smile.compute_smile()
            surface.loc[ttm, smile.strikes] = ivs

        return surface

    def _concat_market_data(self):
        """
        Concatenates strikes, times to maturity, ATM/forward references and market
        prices/ivs across every smile, in smile order. Used to build surface-wide
        (as opposed to per-maturity) cost and evaluation functions.

        :returns: Tuple (strikes, time_to_maturities, spots, market_prices, market_ivs),
            each a flat np.array with one entry per (smile, strike) pair.
        :rtype: tuple
        """
        strikes, maturities, spots, market_prices, market_ivs = [], [], [], [], []
        for smile in self.smiles:
            n = len(smile.strikes)
            strikes.append(np.asarray(smile.strikes))
            maturities.append(np.full(n, smile.time_to_maturity))
            spots.append(np.full(n, smile.atm))
            market_prices.append(np.asarray(smile.market_prices))
            market_ivs.append(np.asarray(smile.market_ivs))

        return (
            np.concatenate(strikes),
            np.concatenate(maturities),
            np.concatenate(spots),
            np.concatenate(market_prices),
            np.concatenate(market_ivs),
        )

    def svi_smooth(self, select_svi_ivs: bool = False):
        """
        Smooths every smile of the surface independently using the raw SVI
        parameterization (see VolatilitySmile.svi_smooth).

        :param bool select_svi_ivs: If True, replaces each smile's market_ivs with
            its calibrated SVI implied volatilities. Default is False.

        :returns: Dictionary of calibrated SVI parameters and dictionary of calibrated
            implied volatilities, both keyed by time to maturity.
        :rtype: Tuple[Dict[float, dict], Dict[float, np.array]]
        """
        calibrated_params = {}
        calibrated_ivs = {}
        for smile in self.smiles:
            params, ivs = smile.svi_smooth(select_svi_ivs=select_svi_ivs)
            calibrated_params[smile.time_to_maturity] = params
            calibrated_ivs[smile.time_to_maturity] = ivs

        return calibrated_params, calibrated_ivs

    def ssvi_smooth(self, select_ssvi_ivs: bool = False):
        """
        Jointly calibrates the SSVI (Surface SVI) parameterization across every
        maturity of the surface at once (see ssvi.py's calibrate_joint): a single
        rho and a single power-law wing function phi(theta) are shared by every
        maturity, tying them together by construction — unlike svi_smooth, which
        still fits each slice independently (see this class's docstring). theta(T)
        (ATM total variance) is read directly off each maturity's own market data,
        not fitted.

        The wing function's exponent and scale are bounded so that the
        Gatheral-Jacquier (2014) sufficient condition for the whole surface to be
        free of calendar-spread and butterfly arbitrage holds by construction (see
        calibrate_joint's docstring for the exact condition).

        :param bool select_ssvi_ivs: If True, replaces each smile's market_ivs with
            its calibrated SSVI implied volatilities. Default is False.

        :returns: Dictionary with the shared 'rho', 'eta', 'gamma', plus per-maturity
            'phi' and 'theta' dictionaries (keyed by time to maturity), and a
            dictionary of calibrated implied volatilities keyed by time to maturity.
        :rtype: Tuple[dict, Dict[float, np.array]]
        """
        ssvi = SSVI(maturities=np.array(self.time_to_maturities))

        forwards = [smile.atm * np.exp(smile.r * smile.time_to_maturity) for smile in self.smiles]
        joint_params = ssvi.calibrate_joint(
            strikes_list=[smile.strikes for smile in self.smiles],
            market_ivs_list=[smile.market_ivs for smile in self.smiles],
            forwards=forwards,
            maturities=self.time_to_maturities,
        )

        calibrated_ivs = {}
        for smile, forward in zip(self.smiles, forwards):
            T = smile.time_to_maturity
            k = np.log(smile.strikes / forward)
            total_variance = ssvi.ssvi_total_variance(
                k, joint_params["theta"][T], joint_params["rho"], joint_params["phi"][T]
            )
            ivs = np.sqrt(total_variance / T)
            calibrated_ivs[T] = ivs

            if select_ssvi_ivs:
                smile.market_ivs = ivs

        return joint_params, calibrated_ivs

    def forward_densities(
        self,
        num_points: int = 500,
        strike_range: tuple[float, float] = None,
        joint_params: dict = None,
    ) -> dict[float, ForwardDensity]:
        """
        Computes the forward risk-neutral density for every maturity of the surface,
        via Breeden-Litzenberger on the surface's jointly-calibrated SSVI fit
        (ssvi_smooth) rather than each smile's own independent SVI fit — the shared
        rho and wing function tie the resulting densities together across
        maturities, consistent with the rest of the surface (see
        VolatilitySmile.forward_density).

        :param int num_points: Number of points in each maturity's strike grid.
        :param tuple strike_range: (K_min, K_max) shared strike grid. If None, each
            smile uses its own market-strike-based range (see forward_density).
        :param dict joint_params: Pre-computed joint SSVI parameters (see ssvi_smooth).
            If None, ssvi_smooth() is run to obtain them. Passing them in avoids
            re-running the (costly) joint calibration when the caller already has them.

        :returns: Dictionary {time_to_maturity: ForwardDensity}.
        :rtype: Dict[float, ForwardDensity]
        """
        if joint_params is None:
            joint_params, _ = self.ssvi_smooth()

        densities = {}
        for smile in self.smiles:
            T = smile.time_to_maturity
            ssvi_params = {
                "rho": joint_params["rho"],
                "theta": joint_params["theta"][T],
                "phi": joint_params["phi"][T],
            }
            densities[T] = smile.forward_density(
                num_points=num_points, strike_range=strike_range, ssvi_params=ssvi_params
            )

        return densities

    def plot_forward_densities(
        self,
        num_points: int = 500,
        strike_range: tuple[float, float] = None,
    ) -> dict[float, ForwardDensity]:
        """
        Plots the forward risk-neutral density for every maturity of the surface,
        overlaid on the same strike axis, to visualize how the distribution's shape
        (skew, kurtosis) evolves with time to maturity.

        :param int num_points: Number of points in each maturity's strike grid.
        :param tuple strike_range: (K_min, K_max) shared strike grid. If None, each
            smile uses its own market-strike-based range (see forward_density).

        :returns: Dictionary {time_to_maturity: ForwardDensity}, as computed for the plot.
        :rtype: Dict[float, ForwardDensity]
        """
        densities = self.forward_densities(num_points=num_points, strike_range=strike_range)
        maturities_sorted = sorted(densities.keys())

        plt.figure(figsize=(9, 6))
        cmap = plt.get_cmap("viridis")
        for i, T in enumerate(maturities_sorted):
            fd = densities[T]
            color = cmap(i / max(len(maturities_sorted) - 1, 1))
            plt.plot(fd.strikes, fd.density, color=color, linewidth=1.5, label=f"T={T:.2f}y")

        plt.xlabel("Strike", fontdict=fontdict)
        plt.ylabel("Density", fontdict=fontdict)
        plt.title("Forward risk-neutral densities across maturities", fontdict=fontdict)
        plt.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)
        plt.legend()
        plt.show()

        return densities

    def plot_forward_density_heatmap(
        self,
        num_points: int = 200,
        num_maturities: int = 100,
        strike_range: tuple[float, float] = None,
    ) -> dict[float, ForwardDensity]:
        """
        Plots the forward risk-neutral density as a 2D heatmap: time to maturity on
        the x-axis, strike on the y-axis, density encoded by color, with the mean,
        median and interquartile range (Q1/Q3) of the underlying overlaid as curves.

        The surface usually only has a handful of quoted maturities, which would
        make the heatmap look blocky along the time axis. Rather than interpolating
        the already-computed densities directly (which would ignore the model and
        risks artifacts — spline overshoot, negative density, drifting mass — between
        maturities), the surface's jointly-calibrated SSVI parameters (rho, eta,
        gamma) are reused as-is, and only theta(T) — the ATM total variance term
        structure — is interpolated between the surface's quoted maturities, via a
        monotone-preserving PCHIP spline (theta must be non-decreasing in T to avoid
        calendar-spread arbitrage). The SSVI density is then recomputed from scratch
        at each fine T from that interpolated theta(T), exactly as
        VolatilitySmile.forward_density does for a real smile.

        All maturities share the same strike grid (needed to build the heatmap), so
        strike_range defaults to the combined market strike range across every smile,
        padded by one span on each side, rather than each smile's own range.

        :param int num_points: Number of points in the shared strike grid.
        :param int num_maturities: Number of interpolated points along the time axis.
        :param tuple strike_range: (K_min, K_max) shared strike grid. If None, defaults
            to the combined market strike range across all smiles, padded by one span.

        :returns: Dictionary {time_to_maturity: ForwardDensity} at the surface's
            actual maturities (not the fine interpolation grid used for the heatmap).
        :rtype: Dict[float, ForwardDensity]
        """
        if strike_range is None:
            all_strikes = np.concatenate([smile.strikes for smile in self.smiles])
            span = all_strikes.max() - all_strikes.min()
            strike_range = (max(1e-6, all_strikes.min() - span), all_strikes.max() + span)

        joint_params, _ = self.ssvi_smooth()
        densities = self.forward_densities(num_points=num_points, strike_range=strike_range, joint_params=joint_params)

        maturities_sorted = np.array(sorted(joint_params["theta"].keys()))
        thetas_sorted = np.array([joint_params["theta"][T] for T in maturities_sorted])
        theta_interp = PchipInterpolator(maturities_sorted, thetas_sorted)

        ssvi = SSVI(maturities=maturities_sorted)
        spot = self.smiles[0].atm
        r = self.smiles[0].r

        strikes_grid = np.linspace(strike_range[0], strike_range[1], num_points)
        density_strikes = strikes_grid[1:-1]
        dK = strikes_grid[1] - strikes_grid[0]
        bs = BlackScholes(spot=spot, r=r, mu=r, volatility=0.02)

        maturities_fine = np.linspace(maturities_sorted.min(), maturities_sorted.max(), num_maturities)
        density_fine = np.zeros((num_maturities, len(density_strikes)))
        means = np.zeros(num_maturities)
        medians = np.zeros(num_maturities)
        q1s = np.zeros(num_maturities)
        q3s = np.zeros(num_maturities)

        for i, T in enumerate(maturities_fine):
            theta_T = float(theta_interp(T))
            phi_T = ssvi.phi_power_law(theta_T, joint_params["eta"], joint_params["gamma"])
            forward_T = spot * np.exp(r * T)
            k = np.log(strikes_grid / forward_T)

            total_variance = ssvi.ssvi_total_variance(k, theta_T, joint_params["rho"], phi_T)
            ivs = np.sqrt(total_variance / T)
            call_prices = bs.call_price(strike=strikes_grid, volatility=ivs, time_to_maturity=T)
            second_derivative = (call_prices[2:] - 2 * call_prices[1:-1] + call_prices[:-2]) / dK**2
            density = np.exp(r * T) * second_derivative
            density_fine[i] = density

            mass = np.trapezoid(density, density_strikes)
            means[i] = np.trapezoid(density_strikes * density, density_strikes) / mass
            cdf = cumulative_trapezoid(density, density_strikes, initial=0) / mass
            q1s[i], medians[i], q3s[i] = np.interp([0.25, 0.5, 0.75], cdf, density_strikes)

        plt.figure(figsize=(9, 6))
        mesh = plt.pcolormesh(maturities_fine, density_strikes, density_fine.T, shading="auto", cmap="viridis")
        plt.colorbar(mesh, label="Density")

        plt.plot(maturities_fine, medians, color="white", linestyle="-", linewidth=1.6, label="Median")
        plt.plot(maturities_fine, means, color="white", linestyle="--", linewidth=1.4, label="Mean")
        plt.plot(maturities_fine, q1s, color="white", linestyle=":", linewidth=1.2, label="Q1 / Q3")
        plt.plot(maturities_fine, q3s, color="white", linestyle=":", linewidth=1.2)

        for T in maturities_sorted:
            plt.axvline(T, color="white", linewidth=0.5, alpha=0.3)

        plt.xlabel("Time to Maturity (Years)", fontdict=fontdict)
        plt.ylabel("Strike", fontdict=fontdict)
        plt.title("Forward risk-neutral density across maturities", fontdict=fontdict)
        plt.legend(loc="upper left", framealpha=0.85)
        plt.show()

        return densities

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
        Calibrates a Heston model (kappa, theta, sigma, rho) or a Bates model (kappa,
        theta, sigma, rho, lambda_jump, mu_J, sigma_J) jointly across every maturity
        of the surface, i.e. a single set of parameters must fit all smiles at once.

        v0 is estimated once from the ATM implied volatility of the shortest-maturity
        smile (best available proxy for the instantaneous variance), exactly as
        VolatilitySmile.calibration does for a single expiry.

        :param callable price_function: Function computing option prices under the
            Heston or Bates model (e.g. heston.call_price).
        :param list initial_guess: Initial parameters, [kappa, theta, sigma, rho] for
            Heston or [kappa, theta, sigma, rho, lambda_jump, mu_J, sigma_J] for Bates.
        :param str guess_correlation_sign: Assumption on the correlation sign
            ('positive', 'negative', 'unknown').
        :param str speed: Calibration method ('local' for fast, 'global' for robust).
        :param str power: Loss function exponentiation ('mse', 'mae', 'rmse').
        :param str method: Local optimization algorithm ("L-BFGS-B", "SLSQP" or "trust-constr").
        :param np.array weights: Weights applied to each (smile, strike) observation.
            If None, uniform weights are used.
        :param bool relative_errors: If True, minimizes relative errors instead of absolute ones.

        :returns: Dictionary containing the calibrated parameters.
        :rtype: dict
        """
        strikes, maturities, spots, market_prices, market_ivs = self._concat_market_data()

        if weights is None:
            weights = 1 / len(strikes)
        else:
            weights = np.asarray(weights) / np.sum(weights)

        if len(initial_guess) == 4:
            model_type = "Heston"
        elif len(initial_guess) == 7:
            model_type = "Bates"
        else:
            raise ValueError(
                "Invalid number of parameters in initial_guess. Must corresponds to 'Heston' or 'Bates'."
            )

        ########################################
        #### estimate v0 from the shortest maturity smile
        ########################################
        shortest_smile = min(self.smiles, key=lambda smile: smile.time_to_maturity)
        index_atm = np.argmin(np.abs(shortest_smile.strikes - shortest_smile.atm))
        vol_initial = shortest_smile.market_ivs[index_atm] ** 2

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
                strike=strikes,
                time_to_maturity=maturities,
                s=spots,
            )

            return difference_function(market_prices, model_prices)

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
                    take_step=CustomStep(model_type),
                    T=2.0,
                )
                print(result.message, result.success)
        else:
            raise ValueError("Invalid speed. Choose either 'local', or 'global'.")

        calibrated_params = _get_calibrated_params(
            optmisation_result=result, vol_initial=vol_initial, model_type=model_type
        )
        self.calibrated_params = calibrated_params
        return calibrated_params

    def evaluate_calibration(
        self, model_values: np.array, metric_type: Literal["price", "iv"] = "price"
    ):
        """
        Evaluates the quality of a surface-wide calibration by calculating RMSE, MSE,
        and MAE either on prices or implied volatilities (IVs).

        model_values must follow the same (smile, strike) ordering as returned by
        _concat_market_data (i.e. smiles in list order, strikes within each smile
        in their given order) — the same ordering used by `calibration`.

        :param np.array model_values: Values estimated by the model (prices or IVs),
            flat array matching _concat_market_data's ordering.
        :param str metric_type: 'price' to compare prices, 'iv' to compare IVs. Default is 'price'.

        :returns: Dictionary containing the absolute and relative error metrics.
        :rtype: Dict[str, float]
        """
        _, _, _, market_prices, market_ivs = self._concat_market_data()

        if metric_type == "price":
            actual_values = market_prices
        elif metric_type == "iv":
            actual_values = market_ivs * 100
            model_values = model_values * 100
        else:
            raise ValueError("metric_type must be either 'price' or 'iv'.")

        diff = actual_values - model_values
        diff_rel = diff / (actual_values + 1e-8)

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

    def plot(self, kind: Literal["surface", "heatmap"] = "surface", num_points: int = 100):
        """
        Plots the volatility surface using cubic spline interpolation and marks actual market points.

        :param kind: 'surface' → 3D plot, 'heatmap' → 2D heatmap.
        :param num_points: Number of points for interpolation grid.
        """
        
        # -----------------------------
        # 1. Collect maturities, strikes, vols
        # -----------------------------
        maturities = []
        all_strikes = set()
        market_points = []

        for smile in self.smiles:
            maturities.append(smile.time_to_maturity)
            all_strikes.update(smile.strikes)
            for s, iv in zip(smile.strikes, smile.market_ivs):
                market_points.append((smile.time_to_maturity, s, iv))

        maturities = np.array(sorted(maturities))
        strikes = np.array(sorted(all_strikes))
        market_points = np.array(market_points)

        # -----------------------------
        # 2. Construct raw IV matrix
        # -----------------------------
        raw_surface = np.full((len(maturities), len(strikes)), np.nan)

        for i, smile in enumerate(self.smiles):
            ivs = smile.compute_smile()
            smile_strikes = smile.strikes

            # cubic interpolation in strike dimension
            f = interp1d(smile_strikes, ivs, kind='cubic', fill_value='extrapolate')
            raw_surface[i, :] = f(strikes)

        # -----------------------------
        # 3. 2D cubic spline interpolation over strikes and maturities
        # -----------------------------
        spline = RectBivariateSpline(maturities, strikes, raw_surface, s=0.01)
        maturities_fine = np.linspace(maturities.min(), maturities.max(), num_points)
        strikes_fine = np.linspace(strikes.min(), strikes.max(), num_points)
        surface_fine = spline(maturities_fine, strikes_fine)

        # -----------------------------
        # 4. 3D Surface plot
        # -----------------------------
        if kind == "surface":
            
            X, Y = np.meshgrid(strikes_fine, maturities_fine)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")

            ax.plot_surface(
                X,
                Y,
                surface_fine,
                cmap="viridis",
                edgecolor="k",
                linewidth=0.3,
                alpha=0.9,
            )

            # scatter points réels
            ax.scatter(
                market_points[:, 1],  # strikes
                market_points[:, 0],  # maturities
                market_points[:, 2],  # IVs
                color="red",
                s=30,
                label="Market IVs",
            )

            ax.set_xlabel("Strike")
            ax.set_ylabel("Time to Maturity (Years)")
            ax.set_zlabel("Implied Volatility")
            ax.set_title("Volatility Surface (Cubic Spline + Market Points)")
            ax.legend()
            plt.show()

        # -----------------------------
        # 5. 2D Heatmap
        # -----------------------------
        elif kind == "heatmap":
            plt.figure(figsize=(10, 6))

            plt.imshow(
                surface_fine,
                aspect="auto",
                cmap="viridis",
                extent=[
                    strikes_fine.min(),
                    strikes_fine.max(),
                    maturities_fine.max(),
                    maturities_fine.min(),
                ],
            )
            plt.colorbar(label="Implied Volatility")

            # scatter points réels
            plt.scatter(
                market_points[:, 1],
                market_points[:, 0],
                color="red",
                s=20,
                label="Market IVs",
            )

            plt.xlabel("Strike")
            plt.ylabel("Time to Maturity (Years)")
            plt.title("Volatility Surface - Heatmap (Cubic Spline + Market Points)")
            plt.legend()
            plt.show()


if __name__ == "__main__":

    from hestonpy.models.heston import Heston

    # Synthetic market: call prices generated from a Heston model with known parameters.
    # Note: plot(kind="surface"/"heatmap") needs at least 4 maturities (RectBivariateSpline
    # default cubic degree requires it), hence 4 maturities below.
    true_params = dict(
        spot=100, vol_initial=0.04, r=0.02,
        kappa=2.0, theta=0.05, drift_emm=0.0, sigma=0.4, rho=-0.6,
    )
    heston = Heston(**true_params)

    maturities = [0.25, 0.5, 0.75, 1.0]
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)

    smiles = [
        VolatilitySmile(
            strikes=strikes,
            time_to_maturity=T,
            atm=100.0,
            market_prices=heston.call_price(strike=strikes, time_to_maturity=T),
            r=0.02,
        )
        for T in maturities
    ]
    surface = VolatilitySurface(smiles)

    print("Implied volatility surface:")
    print(surface.compute_surface(), "\n")

    surface.plot(kind="surface")

    # Joint Heston calibration across every maturity at once
    calibrated = surface.calibration(
        price_function=heston.call_price,
        initial_guess=[1.0, 0.03, 0.3, -0.3],
        guess_correlation_sign="negative",
        speed="local",
    )
    print(f"Calibrated Heston parameters: {calibrated}")
    print(f"True parameters: {true_params}\n")

    # Forward risk-neutral densities across maturities
    surface.plot_forward_densities()
    surface.plot_forward_density_heatmap()
