import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, Literal

def get_options_data(
        symbol: str = 'AAPL',
        flag_option: Literal['call', 'put'] = 'call',
    ):
    """
    Retrieve option data for a given stock symbol.

    Parameters:
    - symbol (str): The stock symbol for which to retrieve option data.
    - flag_option (Literal['call', 'put']): Type of options to retrieve; either 'call' or 'put'.

    Returns:
    - dict (key are maturities) of pd.DataFrame containing:
        - option prices (Call Price or Put Price),
        - strike prices (Strike),
        - bid
        - ask
        - implied vol 
        - volumes (Volume), and
        - time to maturity (Time to Maturity) in financial years
        - maturity
    - spot (float): The spot price of the underlying asset.
    """

    try:
        ticker = yf.Ticker(symbol)
        maturities = ticker.options
        today = datetime.today().date()

        if not maturities:
            print(f"No options traded for {symbol}")
            return None, None, None
        
        options_data = []

        for exp_date in maturities:
            opt_chain = ticker.option_chain(exp_date)
            options = opt_chain.calls if flag_option == "call" else opt_chain.puts

            maturity = datetime.strptime(exp_date, '%Y-%m-%d').date()
            business_days = np.busday_count(today.strftime('%Y-%m-%d'), maturity.strftime('%Y-%m-%d'))
            time_to_maturity = business_days / 252

            if not options.empty:
                options['Time to Maturity'] = time_to_maturity
                options['Maturity'] = maturity
                options_data.append(
                    options[['lastPrice', 'bid', 'ask', 'impliedVolatility', 'strike', 'volume', 'Time to Maturity', 'Maturity']]
                )

        if not options_data:
            return None, None, None

        data = pd.concat(options_data, ignore_index=True).dropna()
        data.columns = [
            f'{flag_option.capitalize()} Price', 'Bid', 'Ask', 'Implied Volatility', 
            'Strike', 'Volume', 'Time to Maturity', 'Maturity'
        ]
        data = data[data['Time to Maturity'] > 0]  # Exclure les options expirées

        # Récupération du prix spot
        history = ticker.history(period="1d")
        spot = history['Close'].iloc[-1] if not history.empty else None

        return data, spot, maturities

    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return None, None, None


def filter_data_for_maturity(data: pd.DataFrame, date: str):
    """
    date must be in the form '%Y-%m-%d'
    """
        
    grouped_dict = {
        maturity.strftime('%Y-%m-%d'): df.reset_index(drop=True)
        for maturity, df in data.groupby("Maturity")
    }

    return grouped_dict[date]