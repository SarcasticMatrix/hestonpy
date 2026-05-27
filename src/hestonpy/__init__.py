# read version from installed package
from importlib.metadata import version
__version__ = version("hestonpy")

from .models import Heston, BlackScholes, Bates
from .models import (
    VolatilitySmile,
    VolatilitySurface,
    StochasticVolatilityInspired,
    SurfaceStochasticVolatilityInspired,
    dichotomie,
    newton_raphson,
    reverse_blackScholes,
    compute_smile,
)
from .option import Option, get_options_data, filter_data_for_maturity
from .portfolio import Portfolio, naive_strategy, time_varying_strategy, optimal_allocate_strategy, run_strategies