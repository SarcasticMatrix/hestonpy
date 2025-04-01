import numpy as np
from numpy import random
from scipy.integrate import quad, quad_vec
from tqdm import tqdm
from typing import Literal

import matplotlib.pyplot as plt
from collections import namedtuple


class Bates:
    """
    Bates model for option pricing with stochastic volatility and jumps.

    This class implements the Bates model, which extends the Heston model by
    incorporating jumps in the underlying asset price. It is used for pricing
    European options with stochastic volatility and jump diffusion.

    :param float spot: The current price of the underlying asset.
    :param float vol_initial: The initial variance of the underlying asset.
    :param float r: The risk-free interest rate.
    :param float kappa: The rate at which the variance reverts to the long-term mean.
    :param float theta: The long-term mean of the variance.
    :param float drift_emm: The market price of risk for the variance process.
    :param float sigma: The volatility of the variance process.
    :param float rho: The correlation between the asset price and its variance.
    :param float lambda_jump: The intensity of jumps.
    :param float mu_J: The mean of the jump size.
    :param float sigma_J: The volatility of the jump size.
    :param int seed: Seed for the random number generator.
    """

    def __init__(
        self, spot: float, vol_initial: float, r: float, kappa: float, theta: float, drift_emm: float, sigma: float, rho: float, lambda_jump: float, mu_J: float, sigma_J: float, seed: int=42,
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

        # Jump parameters
        self.lambda_jump = lambda_jump
        self.mu_J = mu_J
        self.sigma_J = sigma_J

        # World parameters
        self.r = r  # interest rate

        self.seed = seed  # random seed

    def simulate(
        self, 
        time_to_maturity: float = 1, 
        scheme: str = "euler", 
        nbr_points: int = 100, 
        nbr_simulations: int = 1000
    ) -> tuple:
        """
        Simule les prix des actifs et la variance en utilisant le modèle Bates avec des sauts dans les prix.

        :param float time_to_maturity: Temps jusqu'à l'échéance de l'option en années.
        :param str scheme: Schéma de discrétisation à utiliser ('euler' ou 'milstein').
        :param int nbr_points: Nombre de points de temps dans chaque simulation.
        :param int nbr_simulations: Nombre de simulations à effectuer.

        :returns: Les prix simulés des actifs, les variances, le comptage des variances nulles et les sauts.
        :rtype: tuple
        """

        # Initialisation des paramètres de simulation
        dt = time_to_maturity / nbr_points
        S = np.zeros((nbr_simulations, nbr_points + 1))  # Prix des actifs
        V = np.zeros((nbr_simulations, nbr_points + 1))  # Variances
        jump_occurences = np.zeros((nbr_simulations, nbr_points + 1))  # Jumps
        S[:, 0] = self.spot
        V[:, 0] = self.vol_initial

        null_variance = 0

        # Début de la simulation des trajectoires
        for i in range(1, nbr_points + 1):

            # Appliquer le schéma de réflexion pour le processus de variance (s'assurer que la variance est positive)
            V[:, i - 1] = np.abs(V[:, i - 1])

            # Vérifier les variances nulles
            if np.any(V[:, i - 1] == 0):
                null_variance += np.sum(V[i - 1, :] == 0)

            # Générer des variables normales pour les mouvements browniens
            N1 = np.random.normal(loc=0, scale=1, size=nbr_simulations)
            N2 = np.random.normal(loc=0, scale=1, size=nbr_simulations)
            ZV = N1 * np.sqrt(dt)
            ZS = (self.rho * N1 + np.sqrt(1 - self.rho ** 2) * N2) * np.sqrt(dt)

            # Générer les sauts pour chaque instant et chaque simulation
            jump_occurences[:, i] = np.random.poisson(self.lambda_jump * dt, nbr_simulations)
            jumps = jump_occurences[:, i] * (np.exp(np.random.normal(self.mu_J, self.sigma_J, nbr_simulations)) - 1)

            # Mise à jour des prix des actifs avec les sauts
            S[:, i] = (
                S[:, i - 1]
                + (self.r + self.drift_emm * np.sqrt(V[:, i - 1])) * S[:, i - 1] * dt
                + np.sqrt(V[:, i - 1]) * S[:, i - 1] * ZS
                + S[:, i - 1] * jumps
            )

            # Mise à jour de la variance (pas de sauts dans la variance, donc même que le modèle Heston)
            V[:, i] = V[:, i - 1] + (
                self.kappa * (self.theta - V[:, i - 1]) - self.drift_emm * V[:, i - 1]
            ) * dt + self.sigma * np.sqrt(V[:, i - 1]) * ZV

            # Ajustement du schéma de Milstein (optionnel)
            if scheme == "milstein":
                S[:, i] += 1 / 2 * V[:, i - 1] * S[:, i - 1] * (ZS**2 - dt)
                V[:, i] += 1 / 4 * self.sigma**2 * (ZV**2 - dt)

        # Retourner les résultats
        if nbr_simulations == 1:
            S = S.flatten()
            V = V.flatten()
            jump_occurences = jump_occurences.flatten()  # Aplatir les sauts si on a une seule simulation

        return S, V, null_variance, jump_occurences



    def plot_simulation(
        self,
        time_to_maturity: float = 1,
        scheme: str = Literal["euler", "milstein"],
        nbr_points: int = 100,
    ) -> tuple:
        """
        Plot a single simulation of the asset price and variance paths.

        This method simulates and plots the paths of the underlying asset price and its variance
        using the specified discretization scheme.

        :param float time_to_maturity: Time to maturity of the option in years.
        :param str scheme: Discretization scheme to use ('euler' or 'milstein').
        :param int nbr_points: Number of time points in the simulation.

        :returns: Simulated asset prices and variances.
        :rtype: tuple
        """
 
        S, V, _, jumps = self.simulate(
            time_to_maturity=time_to_maturity, 
            scheme=scheme, 
            nbr_points=nbr_points, 
            nbr_simulations=1
        )


        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,8))

        ax1.plot(
            np.linspace(0, time_to_maturity, nbr_points + 1), S, label="Risky asset", color="blue", linewidth=1,
        )

        jump_indices = np.where(jumps != 0)[0]
        jump_indices = jump_indices - 1
        if jump_indices.size > 0:
            ax1.scatter(
                np.linspace(0, time_to_maturity, nbr_points + 1)[jump_indices], 
                S[jump_indices], color="red", label="Jumps", zorder=5, marker='+',s=60
            )


        ax1.set_ylabel("Value [$]", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8,)

        ax2.plot(np.linspace(0, 1, nbr_points + 1),np.sqrt(V),label="Volatility",color="orange",linewidth=1,)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Instantaneous volatility [%]", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(visible=True,which="major",linestyle="--",dashes=(5, 10),color="gray",linewidth=0.5,alpha=0.8,)

        fig.suptitle(f"Bates Model Simulation with {scheme} scheme", fontsize=16)
        plt.tight_layout()
        plt.show()

        return S, V

    def characteristic(self, j: int, **kwargs) -> float:
        """
        Extends the characteristic function to include jumps in the Heston model.

        This method calculates the characteristic function for the Bates model,
        which includes both stochastic volatility and jumps.

        :param int j: Indicator for the characteristic function component (1 or 2).
        :param kwargs: Additional keyword arguments for model parameters.

        :returns: The characteristic function.
        :rtype: float
        """
        vol_initial = kwargs.get("vol_initial", self.vol_initial)

        # Model parameters
        kappa = kwargs.get("kappa", self.kappa)
        theta = kwargs.get("theta", self.theta)
        sigma = kwargs.get("sigma", self.sigma)
        rho = kwargs.get("rho", self.rho)
        drift_emm = kwargs.get("drift_emm", self.drift_emm)

        # Jump parameters
        lambda_jump = kwargs.get("lambda_jump", self.lambda_jump)
        mu_J = kwargs.get("mu_J", self.mu_J)
        sigma_J = kwargs.get("sigma_J", self.sigma_J)

        if j == 1:
            uj = 1 / 2
            bj = kappa + drift_emm - rho * sigma
        elif j == 2:
            uj = -1 / 2
            bj = kappa + drift_emm
        else:
            print("Argument j (int) must be 1 or 2")
            return 0
        a = kappa * theta / sigma**2

        dj = lambda u: np.sqrt((rho * sigma * u * 1j - bj) ** 2 - sigma**2 * (2 * uj * u * 1j - u**2))
        gj = lambda u: (rho * sigma * u * 1j - bj - dj(u)) / (rho * sigma * u * 1j - bj + dj(u))

        Cj = lambda tau, u: self.r * u * tau * 1j + a * (
            (bj - rho * sigma * u * 1j + dj(u)) * tau - 2 * np.log((1 - gj(u) * np.exp(dj(u) * tau)) / (1 - gj(u)))
        )
        Dj = lambda tau, u: (bj - rho * sigma * u * 1j + dj(u)) / sigma**2 * (1 - np.exp(dj(u) * tau)) / (1 - gj(u) * np.exp(dj(u) * tau))

        # Jump component
        char_jump = lambda u, tau: np.exp(lambda_jump * tau * (np.exp(1j * u * mu_J - 0.5 * sigma_J**2 * u**2) - 1))

        return lambda x, v, time_to_maturity, u: (
            np.exp(Cj(time_to_maturity, u) + Dj(time_to_maturity, u) * v + u * x * 1j) * char_jump(u, time_to_maturity)
        )

    def call_price(
            self, 
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
            **kwargs
        ):
        """
        Calculate the price of a European call option using the Bates model.

        This method computes the price of a European call option by leveraging
        the Carr-Madan Fourier pricing method.

        :param np.array strike: The strike price of the option.
        :param np.array time_to_maturity: Time to maturity of the option in years.
        :param np.array s: The current price of the underlying asset.
        :param np.array v: The initial variance of the underlying asset.
        :param kwargs: Additional keyword arguments for model parameters.

        :returns: The price of the call option.
        :rtype: float
        """
        
        price = self.carr_madan_price(
            s=s, 
            v=v,
            strike=strike, 
            time_to_maturity=time_to_maturity,
            **kwargs
        )
        return price

    
    def carr_madan_price(
            self, 
            strike: np.array, 
            time_to_maturity: np.array,
            s: np.array = None,
            v: np.array = None,
            error_boolean: bool = False,
            **kwargs
        ):
        """
        Computes the price of a European call option using the Carr-Madan Fourier pricing method.

        This method employs the Carr-Madan approach, leveraging the characteristic
        function to calculate the option price.

        :param np.array strike: The strike price of the option.
        :param np.array time_to_maturity: Time to maturity of the option in years.
        :param np.array s: The current price of the underlying asset.
        :param np.array v: The initial variance of the underlying asset.
        :param bool error_boolean: Flag to return the error associated with the price.
        :param kwargs: Additional keyword arguments for model parameters.

        :returns: The calculated option price and optionally the error.
        :rtype: float or tuple
        """

        if s is None:
            s = self.spot
        x = np.log(s)
        if v is None:
            v = kwargs.get("vol_initial", self.vol_initial)  # Initial variance
        alpha = 0.3

        price_hat = (
            lambda u: np.exp(-self.r * time_to_maturity)
            / (alpha**2 + alpha - u**2 + u * (2 * alpha + 1) * 1j)
            * self.characteristic(j=2, **kwargs)(x, v, time_to_maturity, u - (alpha + 1) * 1j)
        )

        integrand = lambda u: np.exp(-u * np.log(strike) * 1j) * price_hat(u)

        price = np.real(
            np.exp(-alpha * np.log(strike)) / np.pi * quad_vec(f=integrand, a=0, b=1000)[0]
        )

        if error_boolean:
            error = (
                np.exp(-alpha * np.log(strike)) / np.pi * quad_vec(f=integrand, a=0, b=1000)[1]
            )
            return price, error
        else: 
            return price   



if __name__ == "__main__":

    # Paramètres du modèle Bates
    params = {
        'vol_initial': np.float64(0.04061648109278204),
        'kappa': np.float64(0.6923585666487466),
        'theta': np.float64(1.0728827988086265),
        'drift_emm': 0,
        'sigma': np.float64(1.1861044273587131),
        'rho': np.float64(-0.8549556775813509),
        'lambda_jump': np.float64(4.495056249998712),
        'mu_J': np.float64(-0.05650398032298261),
        'sigma_J': np.float64(0.05000126603766209)
    }
    
    bates_model = Bates(
        spot=5580,        
        r=0.00,          
        **params
    )

    bates_model.plot_simulation(time_to_maturity=1, scheme="milstein", nbr_points=252*12)
