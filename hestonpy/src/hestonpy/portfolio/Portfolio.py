import numpy as np

class Portfolio:
    """
    Abstract base class to model portfolio strategies.

    Attributes:
        r (float): Interest rate.
        dt (float): time step
    """
    def __init__(self, r, dt):
        """
        Initializes the Portfolio with the initial value of the portfolio and interest rate.

        Args:
            r (float): Interest rate.
            dt (float): time step
        """
        self.r = r    # Interest rate
        self.dt = dt
    
    def grow_bank_account(self, bank_account):
        """
        Method to grow the bank with interest.

        Args:
            bank_account (float): Current bank balance.

        Returns:
            float: The updated bank balance.
        """
        return bank_account * np.exp(self.r * self.dt)
    
    def grow_stocks_account(self, stocks_account, S_now, S_previous):
        """
        Method to grow the stocks account with interest.

        Args:
            stocks_account (float): Current stocks account balance.
            S_now (float): Current stock price.
            S_previous (float): Previous stock price.

        Returns:
            float: The updated stocks account balance.
        """
        number_of_stocks = stocks_account / S_previous
        return number_of_stocks * S_now
    
    def back_test(self, S, portfolio0, allocation_strategy):
        """
        Method to back test the optimal portfolio strategy.

        Args:
        - S (np.array): path of the stock
        - portfolio0 (float): value at time 0 of the portfolio
        - allocation_strategy (np.array): allocation strategy, same size as S

        Returns:
            bank_account (array_like): Money in the bank account over time.
            stocks_account (array_like): Money in stocks over time.
        """

        stock_allocation = allocation_strategy[0]
        bank_allocation = 1 - stock_allocation

        bank_account = np.empty_like(S)
        stocks_account = np.empty_like(S)

        stocks_account[0] = portfolio0 * stock_allocation
        bank_account[0] = portfolio0 * bank_allocation

        for t in range(1, len(S)):

            # Update the portfolio
            bank_account[t] = self.grow_bank_account(bank_account[t-1])
            stocks_account[t] = self.grow_stocks_account(stocks_account=stocks_account[t-1], S_now=S[t], S_previous=S[t-1])

            # Update the allocation
            stock_allocation = allocation_strategy[t]
            bank_allocation = 1 - stock_allocation
            total_value = bank_account[t] + stocks_account[t]
            bank_account[t] = total_value * bank_allocation
            stocks_account[t] = total_value * stock_allocation
        
        return bank_account, stocks_account
