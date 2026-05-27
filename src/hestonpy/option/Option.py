from dataclasses import dataclass
from typing import Literal


@dataclass
class Option:
    """
    Represents a vanilla European option contract.

    This is a pure contract specification — it carries no market data (spot, rate)
    and no model. Those live in OptionsBook.

    Attributes:
        flag (Literal['call', 'put']): Type of the option.
        strike (float): Strike price.
        time_to_maturity (float): Time to maturity in years.

    Example:
        >>> opt = Option(flag='call', strike=100.0, time_to_maturity=1.0)
    """

    flag: Literal['call', 'put']
    strike: float
    time_to_maturity: float  # in years


if __name__ == "__main__":

    call = Option(flag='call', strike=105.0, time_to_maturity=0.5)
    put  = Option(flag='put',  strike=95.0,  time_to_maturity=1.0)

    print(call)   # Option(flag='call', strike=105.0, time_to_maturity=0.5)
    print(put)    # Option(flag='put',  strike=95.0,  time_to_maturity=1.0)
