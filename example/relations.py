from hestonpy.models.heston import Heston

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

S0 = 100
V0 = 0.06
r = 0.05
kappa = 1
theta = 0.06
drift_emm = 0.01 
sigma = 0.3
rho = -0.5
T = 1
K = 100

################################################################################
### Theta 
# theta1 = 0.01
# Ks = np.arange(start=20, stop=200, step=1)
# prices_sigma1 = []
# for K in tqdm(Ks):
#     heston = Heston(S0, V0, r, kappa, theta1, drift_emm, sigma, rho, T, K=K)
#     price, error = heston.carr_madan_price()
#     prices_sigma1.append(price)

# theta2 = 0.09
# prices_sigma2 = []
# for K in tqdm(Ks):
#     heston = Heston(S0, V0, r, kappa, theta2, drift_emm, sigma, rho, T, K=K)
#     price, error = heston.carr_madan_price()
#     prices_sigma2.append(price)

# theta3 = 1
# prices_sigma3 = []
# for K in tqdm(Ks):
#     heston = Heston(S0, V0, r, kappa, theta3, drift_emm, sigma, rho, T, K=K)
#     price, error = heston.carr_madan_price()
#     prices_sigma3.append(price)


# plt.figure()
# plt.title(r"Call price as a function of price $\theta$")

# plt.plot(Ks, prices_sigma1, label=rf'$\theta={theta1}$', linewidth=0.7, color='red')
# plt.plot(Ks, prices_sigma2, label=rf'$\theta={theta2}$', linewidth=0.7, color='blue')
# plt.plot(Ks, prices_sigma3, label=rf'$\theta={theta3}$', linewidth=0.7, color='green')

# plt.legend()
# plt.xlabel(r'Strike ($K$)')
# plt.ylabel("Price [€]")
# plt.grid()
# plt.show()



###############################################################################
### Price surface

heston = Heston(S0, V0, r, kappa, theta, drift_emm, sigma, rho, T, K)
heston.price_surface()
