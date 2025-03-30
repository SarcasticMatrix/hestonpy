import numpy as np

_heston_scale = [5/10, 3/10, 1.5/10, 2/10]
_bates_scale = [5/10, 3/10, 1.5/10, 2/10, 0.5, 0.05, 0.1]
class CustomStep:
    """
    Classe permettant de définir une stratégie de saut personnalisée pour l'algorithme de basinhopping.
    Utilise une distribution normale avec un écart-type ajusté en fonction du modèle (Heston ou Bates).
    """

    def __init__(self, model_type):
        if model_type == 'Heston':
            self.scale = _heston_scale
        else: 
            self.scale = _bates_scale

    def __call__(self, x):
        """
        Applique un saut gaussien aux paramètres pour explorer l'espace de recherche.
        """
        return x + np.random.normal(scale=self.scale, size=len(x))  # Sauts gaussiens

def _feller(x):
    """
    Vérifie la condition de Feller : 4 * kappa * theta / sigma**2 - 2 >= 0.
    Cela garantit que le processus de variance ne touche pas zéro.
    """
    return 4 * x[0] * x[1] / x[2]**2 - 2

# Cost function and power, relative_errors parameters
def _generate_difference_function(power:str, relative_errors:bool, weights:np.array):
    """
    Génère une fonction de calcul d'erreur entre les prix de marché et les prix du modèle.
    Permet d'utiliser différentes métriques d'erreur (MSE, RMSE, MAE) et des erreurs relatives ou absolues.
    """
    if not relative_errors:
        difference = lambda market_prices, model_prices : market_prices - model_prices
    else:
        difference = lambda market_prices, model_prices : (market_prices - model_prices) / market_prices
            
    if power == 'mae':
        return lambda market_prices, model_prices : np.sum(weights * np.abs(difference(market_prices, model_prices)))
    elif power == 'rmse':
        return lambda market_prices, model_prices : np.sum(weights * np.sqrt(difference(market_prices, model_prices)**2))
    elif power == 'mse':
        return lambda market_prices, model_prices : np.sum(weights * difference(market_prices, model_prices)**2)
    else: 
        raise ValueError("Invalid power. Choose either 'rmse', 'mae', or 'mse'.")
    

def _get_parameters(model_type:str, params:list):
    """
    Transforme une liste de paramètres en dictionnaire utilisable pour les fonctions de pricing.
    Différencie les modèles Heston et Bates en incluant les sauts dans ce dernier cas.
    """
    if model_type == "Heston":
        kappa, theta, sigma, rho = params
        function_params = {
            "kappa": kappa,
            "theta": theta,
            "drift_emm": 0, 
            "sigma": sigma,
            "rho": rho,
        }
            
    elif model_type == "Bates":
        kappa, theta, sigma, rho, lambda_jump, mu_J, sigma_J = params
        function_params = {
            "kappa": kappa,
            "theta": theta,
            "drift_emm": 0,
            "sigma": sigma,
            "rho": rho,
            "lambda_jump": lambda_jump,
            "mu_J": mu_J,
            "sigma_J": sigma_J,
        }
    return function_params

def _set_bounds(model_type, guess_correlation_sign, initial_guess):
    """
    Définit les bornes des paramètres du modèle pour l'optimisation.
    - Contraint rho selon l'intuition sur son signe.
    - Ajoute des bornes pour les paramètres des sauts dans le modèle de Bates.
    """
    # Bounds of parameters
    bounds = [
        (1e-3, 10),  # kappa 
        (1e-3, 3),   # theta
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

    if model_type == "Heston":
        return bounds
    else:
        jump_parameters_bounds = [(0.01, 10), (-0.5, 0.5), (0.05, 1)] # lambda_jump, mu_J, sigma_J,
        bounds = bounds + jump_parameters_bounds
        return bounds

def _get_calibrated_params(optmisation_result, vol_initial, model_type):
    """
    Formate les résultats de l'optimisation en dictionnaire.
    Affiche les paramètres calibrés en fonction du modèle (Heston ou Bates).
    """
    if model_type == 'Heston':
        calibrated_params = {
            "vol_initial": vol_initial, 
            "kappa": optmisation_result.x[0],
            "theta": optmisation_result.x[1],
            "sigma": optmisation_result.x[2],
            "rho": optmisation_result.x[3],
            "drift_emm": 0,
        }
        print(f"Calibrated parameters: v0={vol_initial:.3f} | kappa={optmisation_result.x[0]:.3f} | theta={optmisation_result.x[1]:.3f} | sigma={optmisation_result.x[2]:.3f} | rho={optmisation_result.x[3]:.3f}\n")

    else: 
        calibrated_params = {
            "vol_initial": vol_initial, 
            "kappa": optmisation_result.x[0],
            "theta": optmisation_result.x[1],
            "drift_emm": 0,
            "sigma": optmisation_result.x[2],
            "rho": optmisation_result.x[3],
            "lambda_jump": optmisation_result.x[4],
            "mu_J": optmisation_result.x[5],
            "sigma_J": optmisation_result.x[6],
        }
        print(f"Calibrated parameters:\n v0={vol_initial:.3f} | kappa={optmisation_result.x[0]:.3f} | theta={optmisation_result.x[1]:.3f} | sigma={optmisation_result.x[2]:.3f} | rho={optmisation_result.x[3]:.3f}  | lambda_jump={optmisation_result.x[4]:.3f}  | mu_J={optmisation_result.x[5]:.3f}  | sigma_J={optmisation_result.x[6]:.3f}\n")

    return calibrated_params


def _callbacks(model_type):
    """
    Retourne une fonction callback pour afficher l'évolution des paramètres durant l'optimisation globale.
    Affiche les paramètres actuels si un meilleur minimum est trouvé.
    """
    def callback(x, f, accepted):

        if accepted:
            to_print = f"Parameters: kappa={x[0]:.3f} | theta={x[1]:.3f} | sigma={x[2]:.3f} | rho={x[3]:.3f}"
            if model_type == 'Bates':
                to_print += f"  | lambda_jump={x[4]:.3f}  | mu_J={x[5]:.3f}  | sigma_J={x[6]:.3f}"

            print("at minimum %.6f accepted %d" % (f, accepted))
            print(to_print, "\n")
    
    return callback