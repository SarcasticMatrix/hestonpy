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
from hestonpy.models.calibration.volatilitySmile import VolatilitySmile

fontdict = {"fontsize": 20, "fontweight": "bold"}

from scipy.optimize import minimize, basinhopping, NonlinearConstraint
from typing import Literal
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


class VolatilitySurface:
    """
    Represents a volatility surface constructed from market prices or implied volatilities.
    """

    def __init__(
        self,
        smiles: list[VolatilitySmile]
    ):
        """
        Initializes the VolatilitySurface object as a list of smiles.
        """
        
        self.smiles = smiles
        time_to_maturities = []
        for smile in self.smiles:
            time_to_maturities.append(smile.time_to_maturity)

    def compute_surface(self):
        """
        Computes implied volatilities from option prices using the Black-Scholes model.
        """

        surface = pd.DataFrame(index=self.time_to_maturities)
        for smile in self.smiles:

            ivs = smile.compute_smile()
            strikes = smile.strikes
            surface[strikes] = ivs
    


    def svi_smooth(self, select_svi_ivs: bool = False):
        pass

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
        pass

    def evaluate_calibration(
        self, model_values: np.array, metric_type: Literal["price", "iv"] = "price"
    ):
        pass

    def plot(self, kind: Literal["surface", "heatmap"] = "surface", num_points: int = 100):
        """
        Plots the volatility surface using cubic spline interpolation and marks actual market points.

        :param kind: 'surface' → 3D plot, 'heatmap' → 2D heatmap.
        :param num_points: Number of points for interpolation grid.
        """
        from scipy.interpolate import interp1d, RectBivariateSpline

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
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
