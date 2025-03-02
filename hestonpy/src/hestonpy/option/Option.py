import numpy as np
from models.heston import Heston
from models.blackScholes import BlackScholes
from typing import Literal

class Option:
    """
    Represents a financial option and provides methods to calculate its price using different models (for now, using BlackScholes and Heston).

    Attributes:
        - flag (Literal['call', 'put']): The type of the option ('call' or 'put').
        - strike (float): The strike price of the option.
        - time_to_maturity (float): The time to maturity of the option (in years).
        - interest (float): The risk-free interest rate (annualized).
        - spot (float): The current spot price of the underlying asset.

    Methods:
        - __init__(self, flag: Literal['call', 'put'], strike: float, time_to_maturity: float, interest: float, spot: float) -> None:
            Initializes an Option object with the specified attributes.

        - price_call(self, flag_model: Literal['heston', 'blackScholes'], vol: float, params: list) -> float:
            Calculates the price of a call option using the specified pricing model.

        - price_put(self, flag_model: Literal['heston', 'blackScholes'], vol: float, params: list) -> float:
            Calculates the price of a put option using the specified pricing model.

        - price(self, flag_model: Literal['heston', 'blackScholes'], vol: float, params: list) -> float:
            Calculates the price of the option based on its type ('call' or 'put') using the specified model.

    Example:
        # Create a call option instance
        call_option = Option(
            flag='call',
            strike=100.0,
            time_to_maturity=1.0,
            interest=0.05,
            spot=100.0,
        )

        # Calculate the price of the call option using the Heston model
        call_price = call_option.price('heston', 0.2, [1.5, 0.04, 0.0, 0.2, -0.5])
        print(f"Call Option Price (Heston Model): {call_price:.2f}")

        # Create a put option instance
        put_option = Option(
            flag='put',
            strike=100.0,
            time_to_maturity=1.0,
            interest=0.05,
            spot=100.0,
        )

        # Calculate the price of the put option using the Black-Scholes model
        put_price = put_option.price('blackScholes', 0.2, [0.05])
        print(f"Put Option Price (Black-Scholes Model): {put_price:.2f}")
        """

    def __init__(
            self, 
            flag: Literal['call', 'put'], 
            strike: float, 
            time_to_maturity: float,
            interest: float,
            spot: float,
        ) -> None:
        """
        Initializes an Option object.

        Parameters:
            - flag (Literal['call', 'put']): The type of the option ('call' or 'put').
            - strike (float): The strike price of the option.
            - time_to_maturity (float): The time to maturity of the option (in years).
            - interest (float): The risk-free interest rate (annualized).
            - spot (float): The current spot price of the underlying asset.
        """
        self.flag = flag
        self.strike = strike
        self.time_to_maturity = time_to_maturity
        self.interest = interest
        self.spot = spot

    def price_call(self, flag_model: Literal['heston', 'blackScholes'], vol:float, params: list) -> float:
        """
        Calculates the price of a call option using the specified model.

        Parameters:
            - flag_model (Literal['heston', 'blackScholes']): The model to use for pricing ('heston' or 'blackScholes').
            - vol (float): volatility, V0 for Heston and sigma for BlackScholes
            - params (list): The parameters required by the chosen model. 
                                a) For 'heston', this should include [kappa, theta, drift_emm, sigma, rho]. 
                                b) For 'blackScholes', this should be [mu].

        Returns:
            float: The price of the call option.
        """
        if flag_model == 'heston':
            heston = Heston(
                S0=self.spot,
                V0=vol,
                r=self.interest,
                T=self.time_to_maturity,
                K=self.strike,
                kappa=params[0],
                theta=params[1],
                drift_emm=params[2],
                sigma=params[3],
                rho=params[4],
           )
            price, _ = heston.carr_madan_price()

        elif flag_model == 'blackScholes':
            blackScholes = BlackScholes(
                initial=self.spot,
                r=self.interest, 
                volatility=vol, 
                T=self.time_to_maturity,
                mu=params[0],
            )
            price = blackScholes.call_price(
                strike=self.strike,
            )
        return price

    def price_put(self, flag_model: Literal['heston', 'blackScholes'], vol:float, params: list) -> float:
        """
        Calculates the price of a put option using the specified model.

        Parameters:
            - flag_model (Literal['heston', 'blackScholes']): The model to use for pricing ('heston' or 'blackScholes').
            - vol (float): volatility, V0 for Heston and sigma for BlackScholes
            - params (list): The parameters required by the chosen model. For 'heston', this should include
                           [kappa, theta, drift_emm, sigma, rho]. For 'blackScholes', this should be [mu].

        Returns:
            float: The price of the put option.
        """
        call_price = self.price_call(flag_model, vol, params)
        price = call_price - self.spot + self.strike * np.exp(- self.interest * self.time_to_maturity)
        return price

    def price(self, flag_model: Literal['heston', 'blackScholes'], vol:float, params: list) -> float:
        """
        Calculates the price of the option based on its type ('call' or 'put') using the specified model.

        Parameters:
            - flag_model (Literal['heston', 'blackScholes']): The model to use for pricing ('heston' or 'blackScholes').
            - vol (float): volatility, V0 for Heston and sigma for BlackScholes
            - params (list): The parameters required by the chosen model. For 'heston', this should include
                           [kappa, theta, drift_emm, sigma, rho]. For 'blackScholes', this should be [mu].

        Returns:
            float: The price of the option.
        """
        if self.flag == 'put':
            price = self.price_put(flag_model, vol, params)
        
        elif self.flag == 'call':
            price = self.price_call(flag_model, vol, params)

        return price
    

if __name__ == "__main__":

    # Create a call option instance
    call_option = Option(
        flag='call',
        strike=100.0,
        time_to_maturity=1.0,
        interest=0.05,
        spot=100.0,
    )

    # Calculate the price of the call option using the Heston model
    call_price = call_option.price('heston', 0.2, [1.5, 0.04, 0.0, 0.2, -0.5])
    print(f"Call Option Price (Heston Model): {call_price:.2f}")

    # Create a put option instance
    put_option = Option(
        flag='put',
        strike=100.0,
        time_to_maturity=1.0,
        interest=0.05,
        spot=100.0,
    )

    # Calculate the price of the put option using the Black-Scholes model
    put_price = put_option.price('blackScholes', 0.2, [0.05])
    print(f"Put Option Price (Black-Scholes Model): {put_price:.2f}")
