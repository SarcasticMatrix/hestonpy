from hestonpy.models.heston import Heston
from hestonpy.models.volatilitySmile import VolatilitySmile, fontdict

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from hestonpy.option.data import get_options_data, filter_data_for_maturity

symbol = '^SPX'
all_market_data, spot, maturities = get_options_data(symbol)
r = 0.00
params = {
    "vol_initial": 0.06,
    "kappa": 1.25,
    "theta": 0.06,
    "drift_emm": 0.00,
    "sigma": 0.6,
    "rho": -0.8,
}

fig, axs = plt.subplots(2, 2)
fig.suptitle(f'Market smiles: {symbol}', fontsize=15, fontweight='bold')

for maturity, ax in zip([maturities[3], maturities[4], maturities[8], maturities[12]], axs.flatten()):

    print(maturity)

    # Getting and filtering data 
    full_market_data = filter_data_for_maturity(all_market_data, maturity)

    time_to_maturity = full_market_data['Time to Maturity'].iloc[0]
    strikes = full_market_data['Strike'].values
    bid_prices = full_market_data["Bid"].values
    ask_prices = full_market_data['Ask'].values
    market_ivs = full_market_data['Implied Volatility'].values
    market_prices = full_market_data['Call Price'].values

    marketVolatilitySmile = VolatilitySmile(
        strikes=strikes,
        time_to_maturity=time_to_maturity,
        atm=spot,
        market_ivs=market_ivs,
        r=r
    )
    market_data = marketVolatilitySmile.filters(full_market_data, select_mid_ivs=True)

    # Calibration
    heston = Heston(spot=spot, r=r, **params)
    initial_params = marketVolatilitySmile.calibration(
        price_function=heston.call_price,
        guess_correlation_sign='negative',
        initial_guess=[params['kappa'], params['theta'], params['sigma'], params['rho']],
        speed='local',
    )

    calibrated_params = marketVolatilitySmile.calibration(
        relative_errors=False,
        price_function=heston.call_price,
        guess_correlation_sign='negative',
        initial_guess=[initial_params['kappa'], initial_params['theta'], initial_params['sigma'], initial_params['rho']],
        speed='global',
    )

    # Compute calibrated ivs
    calibrated_prices = heston.call_price(
        strike=marketVolatilitySmile.strikes, time_to_maturity=time_to_maturity, **calibrated_params
    )


    # Some plots
    calibrated_ivs = marketVolatilitySmile.compute_smile(prices=calibrated_prices)
    bid_ivs = market_data['Ask ivs'].values
    ask_ivs = market_data['Bid ivs'].values

    forward = marketVolatilitySmile.atm * np.exp(marketVolatilitySmile.r * marketVolatilitySmile.time_to_maturity)

    ax.axvline(1, linestyle="--", color="gray", label="ATM Strike")

    if ax == axs.flatten()[-1]:
        ax.plot(marketVolatilitySmile.strikes / forward, calibrated_ivs, label="calibred", marker='+', color='blue', linestyle="dotted", markersize=4)
        ax.scatter(marketVolatilitySmile.strikes / forward, marketVolatilitySmile.market_ivs, label="mid", marker='o', color='red', s=20)
        ax.scatter(marketVolatilitySmile.strikes / forward, bid_ivs, label="bid", marker=7, color='black', s=20)
        ax.scatter(marketVolatilitySmile.strikes / forward, ask_ivs, label="ask", marker=6, color='gray', s=20)
    else:
        ax.plot(marketVolatilitySmile.strikes / forward, calibrated_ivs, marker='+', color='blue', linestyle="dotted", markersize=4)
        ax.scatter(marketVolatilitySmile.strikes / forward, marketVolatilitySmile.market_ivs, marker='o', color='red', s=20)
        ax.scatter(marketVolatilitySmile.strikes / forward, bid_ivs, marker=7, color='black', s=20)
        ax.scatter(marketVolatilitySmile.strikes / forward, ask_ivs, marker=6, color='gray', s=20)

    ax.set_xlabel("Moneyness [%]", fontdict=fontdict)
    ax.set_ylabel("Implied Volatility [%]", fontdict=fontdict)

    date = datetime.strptime(maturity, '%Y-%m-%d').date().strftime("%d-%B-%y")
    title = f"{date}: {marketVolatilitySmile.time_to_maturity * 252 / 5:.1f} semaines"
    ax.set_title(title, fontdict=fontdict)
    ax.grid(visible=True, which="major", linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5, alpha=0.8)

fig.legend(loc='lower center', ncol=5) #bbox_to_anchor=(0.5, -0.05), 
plt.tight_layout()
plt.show()
