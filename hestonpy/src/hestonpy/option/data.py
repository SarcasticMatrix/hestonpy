import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Literal
from typing import Tuple, Literal, Optional

def get_options_data(
        symbol: str,
        flag_option: Literal['call', 'put'],
    ) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Retrieve option data for a given stock symbol.

    Parameters:
    - symbol (str): The stock symbol for which to retrieve option data.
    - flag_option (Literal['call', 'put']): Type of options to retrieve; either 'call' or 'put'.

    Returns:
    - pd.DataFrame: A DataFrame containing:
        - option prices (Call Price or Put Price),
        - strike prices (Strike),
        - volumes (Volume), and
        - time to maturity (Time to Maturity) in financial years.
    - spot (float): The spot price of the underlying asset.
    """
    # to do: implémenter les put 

    ticker = yf.Ticker(symbol)
    expiration_dates = ticker.options
    today = datetime.today().date()
    data = []
    
    if expiration_dates != []:
        for exp_date in expiration_dates:
            opt_chain = ticker.option_chain(exp_date)

            if flag_option == "call":
                options = opt_chain.calls
            elif flag_option == 'put':
                options = opt_chain.puts
            else: 
                print("flag_option should be 'call' or 'put'")

            expiration = datetime.strptime(exp_date, '%Y-%m-%d').date()
            time_to_maturity = (expiration - today).days / 252

            for _, row in options.iterrows():
                data.append([row['lastPrice'], row['strike'], row['volume'], time_to_maturity, expiration])

        df = pd.DataFrame(data, columns=[f'{flag_option.capitalize()} Price', 'Strike', 'Volume', 'Time to Maturity', "Expiration Date"])
        df = df.dropna()
        mask = df['Time to Maturity'] != 0.0
        df = df.loc[mask]

        history = ticker.history(period="1d")
        spot = history['Close'].iloc[-1]

        return df, spot
    else:
        print(f"No options traded for {symbol}")