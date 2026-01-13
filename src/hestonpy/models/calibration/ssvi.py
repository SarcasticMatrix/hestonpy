import numpy as np
from scipy.optimize import minimize, Bounds
import warnings

class SurfaceStochasticVolatilityInspired:
    """
    Implements the SSVI (Surface Stochastic Volatility Inspired) parameterization.
    Calibrates the SSVI parameters (rho(T), phi(T)) for each maturity T of a vol surface.

    rho(T) = ATM total implied variance = market_iv_atm(T)^2 * T
    phi(T) is the curvature / 'wing' parameter.
    """

    def __init__(self, maturities: np.array):
        """
        :param maturities: Array of maturities in years.
        """
        self.maturities = maturities
        self.params_surface = {}   # will contain {T: {"rho": ..., "phi": ...}}
        self.theta = {}            # theta(T) total variance ATM per maturity

    # ----------------------------------------------------------------------
    # Raw SSVI formula
    # ----------------------------------------------------------------------
    def ssvi_total_variance(self, k, theta, rho, phi):
        """
        Computes total implied variance w(k,T) using the SSVI formula.

        :param k: log-moneyness
        :param theta: total ATM variance for maturity T
        :param rho: correlation parameter (|rho| < 1)
        :param phi: wing parameter φ(T) >= 0 and theta*phi <= 2

        :returns: total implied variance w(k,T)
        """
        term1 = rho * phi * k
        sqrt_term = np.sqrt((phi * k + rho)**2 + 1 - rho**2)
        return 0.5 * theta * (1 + term1 + sqrt_term)

    # ----------------------------------------------------------------------
    # Calibration for a single maturity
    # ----------------------------------------------------------------------
    def calibrate_single_maturity(self, strikes, market_ivs, forward, T, x0=[0.0, 0.3]):
        """
        Calibrates rho(T), phi(T) for a single maturity.

        :param strikes: strike array
        :param market_ivs: implied volatilities for that maturity
        :param forward: forward price F
        :param T: maturity in years
        :param x0: initial guess [rho, phi]

        :returns: (rho, phi)
        """

        # Compute θ(T) from ATM IV
        atm_iv = market_ivs[np.argmin(np.abs(strikes - forward))]
        theta_T = atm_iv**2 * T
        self.theta[T] = theta_T

        k = np.log(strikes / forward)
        market_tiv = market_ivs**2 * T

        def cost(x):
            rho, phi = x
            model = self.ssvi_total_variance(k, theta_T, rho, phi)
            return np.sum((model - market_tiv)**2)

        # No arbitrage constraints
        bounds = Bounds(
            [-0.999, 1e-6],       # rho, phi
            [0.999, 2/theta_T]    # |rho|<1 , phi <= 2/theta(T)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(cost, x0, method="SLSQP", bounds=bounds)

        rho_opt, phi_opt = result.x
        return rho_opt, phi_opt

    # ----------------------------------------------------------------------
    # Calibrate entire surface
    # ----------------------------------------------------------------------
    def calibrate_surface(self, strikes, iv_surface, forwards):
        """
        Calibrate SSVI for all maturities.

        :param strikes: 1D array of strikes (same for all maturities)
        :param iv_surface: 2D array iv_surface[T_index, strike_index]
        :param forwards: 1D array forward prices F(T)

        :returns: dictionary of calibrated parameters per maturity
        """
        for i, T in enumerate(self.maturities):
            market_ivs = iv_surface[i]
            F = forwards[i]

            rho, phi = self.calibrate_single_maturity(
                strikes=strikes,
                market_ivs=market_ivs,
                forward=F,
                T=T
            )

            self.params_surface[T] = {"rho": rho, "phi": phi}

        return self.params_surface

    # ----------------------------------------------------------------------
    # Compute volatility surface from calibrated parameters
    # ----------------------------------------------------------------------
    def compute_iv_surface(self, strikes, forwards):
        """
        Generates the implied vol surface from calibrated SSVI params.

        :param strikes: 1D array
        :param forwards: 1D array forward for each maturity

        :returns: 2D array of implied volatilities
        """
        iv_surface = np.zeros((len(self.maturities), len(strikes)))
        for i, T in enumerate(self.maturities):
            rho = self.params_surface[T]["rho"]
            phi = self.params_surface[T]["phi"]
            theta_T = self.theta[T]

            k = np.log(strikes / forwards[i])
            tiv = self.ssvi_total_variance(k, theta_T, rho, phi)
            iv_surface[i] = np.sqrt(tiv / T)

        return iv_surface
