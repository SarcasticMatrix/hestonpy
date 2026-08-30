import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds
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
        self.params_surface = {}   # per-maturity fit: {T: {"rho": ..., "phi": ...}}
        self.joint_params = {}     # joint fit: {"rho", "eta", "gamma", "phi": {T:...}, "theta": {T:...}}
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
        phi_upper = 2 / theta_T
        bounds = Bounds(
            [-0.999, 1e-6],       # rho, phi
            [0.999, phi_upper]    # |rho|<1 , phi <= 2/theta(T)
        )

        # A single local SLSQP run from x0 tends to get stuck immediately at x0
        # (the rho/phi scales are very different, and phi's bound width varies a
        # lot with theta(T)): basinhopping with a per-parameter step scale, mirroring
        # StochasticVolatilityInspired.calibration's own use of basinhopping, avoids
        # that degenerate local optimum.
        step_scale = np.array([0.4, max(0.3, 0.2 * phi_upper)])

        def take_step(x):
            return x + np.random.normal(scale=step_scale, size=len(x))

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = basinhopping(
                cost,
                x0=x0,
                niter=200,
                niter_success=30,
                take_step=take_step,
                minimizer_kwargs=minimizer_kwargs,
            )

        rho_opt, phi_opt = result.x
        return rho_opt, phi_opt

    # ----------------------------------------------------------------------
    # Shared power-law wing function, for a joint fit across every maturity
    # ----------------------------------------------------------------------
    @staticmethod
    def phi_power_law(theta, eta, gamma):
        """
        Gatheral's power-law parameterization of the SSVI wing function phi(theta):
        phi(theta) = eta / (theta**gamma * (1+theta)**(1-gamma)).

        Unlike calibrate_single_maturity's free phi(T) (one independent value per
        maturity), this phi is a single function of theta shared by every maturity,
        which is what ties the whole surface together in calibrate_joint.

        :param theta: ATM total variance theta(T) (scalar or array).
        :param eta: Overall scale of the wings.
        :param gamma: Decay exponent, in (0, 1).

        :returns: phi(theta)
        """
        return eta / (theta**gamma * (1 + theta)**(1 - gamma))

    # ----------------------------------------------------------------------
    # Joint calibration across every maturity at once
    # ----------------------------------------------------------------------
    def calibrate_joint(self, strikes_list, market_ivs_list, forwards, maturities, x0=[0.0, 0.5, 0.3], niter=300):
        """
        Jointly calibrates a single rho and a single power-law wing function
        phi(theta; eta, gamma) across every maturity at once, instead of fitting
        rho(T)/phi(T) independently per maturity as calibrate_single_maturity does.
        theta(T) is still read directly off each maturity's own ATM implied
        volatility, not fitted.

        Free parameters are [rho, u, gamma], reparameterized so that:
          - rho in (-1, 1): correlation, shared across every maturity.
          - gamma in (0, 0.5]: Gatheral & Jacquier (2014, "Arbitrage-free SVI
            volatility surfaces") show this range, together with the next point,
            is a *sufficient* condition for the whole surface to be free of both
            calendar-spread and butterfly arbitrage.
          - u in (0, 1], with eta = u * 2 / (1 + |rho|), which guarantees
            eta * (1 + |rho|) <= 2 for any u -- the other half of that sufficient
            condition. Reparameterizing this way turns what would otherwise be a
            nonlinear constraint into a plain box constraint on u, so it is
            enforced by construction rather than checked after the fact, and keeps
            the same bounded-optimizer approach that fixed
            calibrate_single_maturity's SLSQP-stuck-at-x0 issue.

        :param strikes_list: list of strike arrays, one per maturity (same order as maturities).
        :param market_ivs_list: list of market implied vol arrays, one per maturity.
        :param forwards: list/array of forward prices, one per maturity.
        :param maturities: list/array of maturities in years, same order as the above.
        :param x0: initial guess [rho, u, gamma].
        :param niter: number of basinhopping iterations.

        :returns: dict with the shared 'rho', 'eta', 'gamma', plus per-maturity
            'theta' and 'phi' dictionaries (keyed by maturity).
        :rtype: dict
        """
        maturities = list(maturities)
        thetas, ks, market_tivs = {}, {}, {}

        for strikes, market_ivs, forward, T in zip(strikes_list, market_ivs_list, forwards, maturities):
            atm_iv = market_ivs[np.argmin(np.abs(strikes - forward))]
            thetas[T] = atm_iv**2 * T
            ks[T] = np.log(strikes / forward)
            market_tivs[T] = market_ivs**2 * T

        def cost(x):
            rho, u, gamma = x
            eta = u * 2 / (1 + abs(rho))
            total = 0.0
            for T in maturities:
                phi_T = self.phi_power_law(thetas[T], eta, gamma)
                model = self.ssvi_total_variance(ks[T], thetas[T], rho, phi_T)
                total += np.sum((model - market_tivs[T])**2)
            return total

        bounds = Bounds([-0.999, 1e-3, 1e-3], [0.999, 1.0, 0.5])
        step_scale = np.array([0.4, 0.3, 0.15])

        def take_step(x):
            return x + np.random.normal(scale=step_scale, size=len(x))

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = basinhopping(
                cost,
                x0=x0,
                niter=niter,
                niter_success=40,
                take_step=take_step,
                minimizer_kwargs=minimizer_kwargs,
            )

        rho_opt, u_opt, gamma_opt = result.x
        eta_opt = u_opt * 2 / (1 + abs(rho_opt))

        self.theta = thetas
        phi_per_maturity = {T: self.phi_power_law(thetas[T], eta_opt, gamma_opt) for T in maturities}
        self.joint_params = {
            "rho": rho_opt,
            "eta": eta_opt,
            "gamma": gamma_opt,
            "phi": phi_per_maturity,
            "theta": thetas,
        }
        return self.joint_params

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
