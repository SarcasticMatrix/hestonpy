# hestonpy

`hestonpy` is a Python package for pricing, calibrating and hedging options under stochastic volatility models, built around the Heston framework (with Black-Scholes and Bates also included). It's aimed at anyone who wants to simulate paths, price vanilla options, calibrate a model to a market smile, or explore optimal portfolio allocation under Heston dynamics, without having to reimplement the underlying numerical machinery from scratch.

Documentation: https://sarcasticmatrix.github.io/hestonpy/

## Installation

`hestonpy` is available on PyPI:

```bash
pip install hestonpy
```

It requires Python 3.10+ and depends on `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm` and `yfinance`.

## Features

- **Models**: Black-Scholes, Heston, and Bates (Heston with jumps)
- **Simulation**: Euler and Milstein discretization schemes for asset and variance paths
- **Pricing**: Monte Carlo, Fourier-transform, and Carr-Madan methods for European vanilla options
- **Greeks & hedging**: delta/vega computation and delta-vega hedging
- **Calibration**: fit model parameters to market implied volatility smiles and surfaces, with support for pulling option chain data from Yahoo Finance or user-supplied data
- **SVI / SSVI**: Stochastic Volatility Inspired parametrization for smiles and surfaces

## Quick start

```python
from hestonpy import Heston

model = Heston(
    spot=100,
    vol_initial=0.04,
    r=0.02,
    kappa=2.0,
    theta=0.04,
    drift_emm=0.0,
    sigma=0.3,
    rho=-0.7,
)

# Simulate asset price and variance paths
S, V, null_variance = model.simulate(
    time_to_maturity=1,
    scheme="milstein",
    nbr_points=252,
    nbr_simulations=10_000,
)

# Price a European call
price = model.call_price(strike=100, time_to_maturity=1)
```

Calibrating to a market smile:

```python
from hestonpy import VolatilitySmile

smile = VolatilitySmile(...)  # see docs for constructing from market/Yahoo Finance data
calibrated_params = smile.calibration(...)
```

More worked examples (pricing, calibration, hedging, asset allocation) are available in the `example/` directory of the repository.

## Project layout

```
src/hestonpy/
├── models/         # Black-Scholes, Heston, Bates, calibration (SVI/SSVI)
├── option/         # Option and OptionsBook abstractions, market data fetching
```

## Contributing

Contributions are welcome — see `CONTRIBUTING.md` and `CONDUCT.md` for guidelines.

## License

`hestonpy` was created by Théophile Schmutz ([@SarcasticMatrix](https://github.com/SarcasticMatrix)). It is licensed under the MIT license — see `LICENSE` for details.