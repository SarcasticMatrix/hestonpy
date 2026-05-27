from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from hestonpy.option.Option import Option
from hestonpy.models.heston import Heston
from hestonpy.models.blackScholes import BlackScholes


@dataclass
class Position:
    """
    A position in the options book: a contract, a signed quantity, and an optional entry price.

    Attributes:
        option (Option): The option contract.
        quantity (float): Signed quantity — positive = long, negative = short.
        entry_price (float | None): Price at which the position was opened. Used for P&L.
    """

    option: Option
    quantity: float
    entry_price: float | None = None


class OptionsBook:
    """
    Manages a book of European vanilla options on a single underlying.

    Market context (spot, risk-free rate) is held at the book level and shared
    across all positions. Pricing and greeks delegate to the existing BlackScholes
    and Heston model classes.

    Greeks are always computed analytically via Black-Scholes. Pricing can use
    either Black-Scholes or Heston, specified at call time.

    Parameters:
        spot (float): Current spot price of the underlying.
        r (float): Risk-free interest rate (annualised).
        underlying (str): Optional label for the underlying (e.g. 'AAPL').

    Example:
        >>> book = OptionsBook(spot=100.0, r=0.05, underlying='SPX')
        >>> book.add('call', strike=105.0, time_to_maturity=0.5, quantity=1)
        >>> book.add('put',  strike=95.0,  time_to_maturity=0.5, quantity=-2)
        >>> print(book)
        >>> print(book.summary('blackScholes', vol=0.2, params=[0.05]))
    """

    def __init__(self, spot: float, r: float, underlying: str = ""):
        self.spot = spot
        self.r = r
        self.underlying = underlying
        self._positions: list[Position] = []

    # ──────────────────────────────────────────────────────────────
    # Book management
    # ──────────────────────────────────────────────────────────────

    def add(
        self,
        flag: Literal['call', 'put'],
        strike: float,
        time_to_maturity: float,
        quantity: float,
        entry_price: float | None = None,
    ) -> None:
        """
        Add a new position to the book.

        Parameters:
            flag: 'call' or 'put'.
            strike: Strike price.
            time_to_maturity: Time to maturity in years.
            quantity: Signed quantity (+N = long, -N = short).
            entry_price: Optional entry price for P&L tracking.
        """
        option = Option(flag=flag, strike=strike, time_to_maturity=time_to_maturity)
        self._positions.append(Position(option=option, quantity=quantity, entry_price=entry_price))

    def remove(self, index: int) -> None:
        """Remove the position at the given index."""
        self._positions.pop(index)

    def update_spot(self, new_spot: float) -> None:
        """Update the spot price (re-marks the book to the new level)."""
        self.spot = new_spot

    # ──────────────────────────────────────────────────────────────
    # Internal pricing
    # ──────────────────────────────────────────────────────────────

    def _price_option(
        self,
        option: Option,
        flag_model: Literal['heston', 'blackScholes'],
        vol: float,
        params: list,
    ) -> float:
        """
        Price a single option using the chosen model.

        Parameters:
            option: The Option contract to price.
            flag_model: 'blackScholes' or 'heston'.
            vol: Volatility — sigma for BS, V0 for Heston.
            params:
                - BlackScholes: [mu]
                - Heston:       [kappa, theta, drift_emm, sigma, rho]

        Returns:
            float: Option price.
        """
        if flag_model == 'blackScholes':
            bs = BlackScholes(
                spot=self.spot,
                r=self.r,
                mu=params[0],
                volatility=vol,
            )
            if option.flag == 'call':
                return bs.call_price(strike=option.strike, time_to_maturity=option.time_to_maturity)
            else:
                return bs.put_price(strike=option.strike, time_to_maturity=option.time_to_maturity)

        elif flag_model == 'heston':
            heston = Heston(
                spot=self.spot,
                vol_initial=vol,
                r=self.r,
                kappa=params[0],
                theta=params[1],
                drift_emm=params[2],
                sigma=params[3],
                rho=params[4],
            )
            call_price = heston.carr_madan_price(
                strike=option.strike,
                time_to_maturity=option.time_to_maturity,
            )
            if option.flag == 'call':
                return call_price
            else:
                # Put via put-call parity
                return call_price - self.spot + option.strike * np.exp(-self.r * option.time_to_maturity)

        else:
            raise ValueError(f"Unknown model '{flag_model}'. Choose 'blackScholes' or 'heston'.")

    # ──────────────────────────────────────────────────────────────
    # Book-level pricing & P&L
    # ──────────────────────────────────────────────────────────────

    def price(
        self,
        flag_model: Literal['heston', 'blackScholes'],
        vol: float,
        params: list,
    ) -> float:
        """
        Total mark-to-market value of the book.

        Returns:
            float: Sum of (quantity × price) across all positions.
        """
        return sum(
            pos.quantity * self._price_option(pos.option, flag_model, vol, params)
            for pos in self._positions
        )

    def pnl(
        self,
        flag_model: Literal['heston', 'blackScholes'],
        vol: float,
        params: list,
    ) -> float:
        """
        Total P&L of the book vs entry prices.

        Positions without an entry_price are excluded from the calculation.

        Returns:
            float: Sum of quantity × (current_price - entry_price).
        """
        total = 0.0
        for pos in self._positions:
            if pos.entry_price is not None:
                current = self._price_option(pos.option, flag_model, vol, params)
                total += pos.quantity * (current - pos.entry_price)
        return total

    # ──────────────────────────────────────────────────────────────
    # Greeks (Black-Scholes analytical)
    # ──────────────────────────────────────────────────────────────

    def greeks(self, vol: float) -> dict[str, float]:
        """
        Aggregate Black-Scholes greeks of the book.

        Delta, Gamma and Vega are computed analytically from the BS closed-form
        formulas, regardless of which model is used for pricing. This is the
        standard market convention.

        Parameters:
            vol: Implied volatility to use for greek computation.

        Returns:
            dict with keys 'delta', 'gamma', 'vega'.
        """
        bs = BlackScholes(spot=self.spot, r=self.r, mu=self.r, volatility=vol)

        delta_total = sum(
            pos.quantity * bs.delta(
                strike=pos.option.strike,
                time_to_maturity=pos.option.time_to_maturity,
                flag_option=pos.option.flag,
            )
            for pos in self._positions
        )
        gamma_total = sum(
            pos.quantity * bs.gamma(
                strike=pos.option.strike,
                time_to_maturity=pos.option.time_to_maturity,
            )
            for pos in self._positions
        )
        vega_total = sum(
            pos.quantity * bs.vega(
                strike=pos.option.strike,
                time_to_maturity=pos.option.time_to_maturity,
            )
            for pos in self._positions
        )

        return {
            'delta': round(delta_total, 6),
            'gamma': round(gamma_total, 6),
            'vega':  round(vega_total,  6),
        }

    # ──────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────

    def summary(
        self,
        flag_model: Literal['heston', 'blackScholes'],
        vol: float,
        params: list,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per position, plus a TOTAL row.

        Columns: flag, strike, TTM, qty, price, MtM, PnL, delta, gamma, vega.

        Parameters:
            flag_model: 'blackScholes' or 'heston'.
            vol: Volatility used for both pricing and greek computation.
            params: Model parameters (see _price_option).
        """
        bs = BlackScholes(spot=self.spot, r=self.r, mu=self.r, volatility=vol)
        rows = []

        for pos in self._positions:
            opt        = pos.option
            unit_price = self._price_option(opt, flag_model, vol, params)
            unit_delta = bs.delta(strike=opt.strike, time_to_maturity=opt.time_to_maturity, flag_option=opt.flag)
            unit_gamma = bs.gamma(strike=opt.strike, time_to_maturity=opt.time_to_maturity)
            unit_vega  = bs.vega(strike=opt.strike,  time_to_maturity=opt.time_to_maturity)
            pnl_pos    = pos.quantity * (unit_price - pos.entry_price) if pos.entry_price is not None else None

            rows.append({
                'flag':   opt.flag,
                'strike': opt.strike,
                'TTM':    opt.time_to_maturity,
                'qty':    pos.quantity,
                'price':  round(unit_price,                4),
                'MtM':    round(pos.quantity * unit_price, 4),
                'PnL':    round(pnl_pos,                   4) if pnl_pos is not None else '-',
                'delta':  round(pos.quantity * unit_delta, 6),
                'gamma':  round(pos.quantity * unit_gamma, 6),
                'vega':   round(pos.quantity * unit_vega,  6),
            })

        df = pd.DataFrame(rows)
        df.index.name = '#'

        # TOTAL: only aggregate positions that have an entry price
        tracked = df[df['PnL'] != '-']

        def _sum(col, decimals):
            return round(tracked[col].sum(), decimals) if not tracked.empty else '-'

        totals = pd.Series({
            'flag':   '',
            'strike': '',
            'TTM':    '',
            'qty':    tracked['qty'].sum()      if not tracked.empty else '-',
            'price':  '',
            'MtM':    _sum('MtM',   4),
            'PnL':    _sum('PnL',   4),
            'delta':  _sum('delta', 6),
            'gamma':  _sum('gamma', 6),
            'vega':   _sum('vega',  6),
        }, name='TOTAL')

        return pd.concat([df, totals.to_frame().T])

    # ──────────────────────────────────────────────────────────────
    # Dunder
    # ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        header = (
            f"\nOptionsBook  underlying={self.underlying}  spot={self.spot}  r={self.r}  "
            f"({len(self)} position{'s' if len(self) != 1 else ''})"
        )
        if not self._positions:
            return header + "\n  (empty)"

        # Pre-format every field so we can measure true widths
        indices      = [f"[{i}]"                                             for i, _   in enumerate(self._positions)]
        qtys         = [f"+{p.quantity}" if p.quantity >= 0 else str(p.quantity) for p in self._positions]
        types        = [p.option.flag                                         for p in self._positions]
        strikes      = [str(p.option.strike)                                  for p in self._positions]
        ttms         = [f"{p.option.time_to_maturity:.3f}y"                   for p in self._positions]
        entry_prices = [str(p.entry_price) if p.entry_price is not None else "-" for p in self._positions]

        # Column widths = max of header label vs any value
        w_idx    = max(len("   "), max(len(s) for s in indices))
        w_qty    = max(len("qty"),    max(len(s) for s in qtys))
        w_type   = max(len("type"),   max(len(s) for s in types))
        w_strike = max(len("strike"), max(len(s) for s in strikes))
        w_ttm    = max(len("TTM"),    max(len(s) for s in ttms))
        w_entry  = max(len("entry"),  max(len(s) for s in entry_prices))

        sep = "  "
        header_row = (
            f"  {'':>{w_idx}}{sep}"
            f"{'qty':>{w_qty}}{sep}"
            f"{'type':<{w_type}}{sep}"
            f"{'strike':>{w_strike}}{sep}"
            f"{'TTM':>{w_ttm}}{sep}"
            f"{'entry':>{w_entry}}"
        )
        divider = "  " + "-" * (len(header_row) - 2)

        lines = [header, "", header_row, divider]
        for idx, qty, typ, strike, ttm, entry in zip(indices, qtys, types, strikes, ttms, entry_prices):
            lines.append(
                f"  {idx:>{w_idx}}{sep}"
                f"{qty:>{w_qty}}{sep}"
                f"{typ:<{w_type}}{sep}"
                f"{strike:>{w_strike}}{sep}"
                f"{ttm:>{w_ttm}}{sep}"
                f"{entry:>{w_entry}}"
            )
        return '\n'.join(lines)


if __name__ == "__main__":

    # ── Création du book ─────────────────────────────────────────
    book = OptionsBook(spot=100.0, r=0.05, underlying='SPX')

    # Long 1 call OTM, short 2 puts OTM (risk-reversal-like), long 1 call lointain
    book.add('call', strike=105.0, time_to_maturity=0.5, quantity= 1, entry_price=3.5)
    book.add('put',  strike=95.0,  time_to_maturity=0.5, quantity=-2, entry_price=2.1)
    book.add('call', strike=110.0, time_to_maturity=1.0, quantity= 1)

    print(book)
    print()

    # ── Pricing & P&L sous Black-Scholes ─────────────────────────
    vol = 0.2
    bs_params = [0.05]   # [mu]

    print("=== Black-Scholes ===")
    print(f"MtM    : {book.price('blackScholes', vol=vol, params=bs_params):.4f}")
    print(f"PnL    : {book.pnl(  'blackScholes', vol=vol, params=bs_params):.4f}")
    print(f"Greeks : {book.greeks(vol=vol)}")
    print()
    print(book.summary('blackScholes', vol=vol, params=bs_params).to_string())
    print()

    # ── Pricing & P&L sous Heston ─────────────────────────────────
    # params = [kappa, theta, drift_emm, sigma, rho]
    heston_params = [1.5, 0.04, 0.0, 0.2, -0.5]
    vol_initial   = 0.04   # V0 (variance initiale)

    print("=== Heston ===")
    print(f"MtM    : {book.price('heston', vol=vol_initial, params=heston_params):.4f}")
    print(f"PnL    : {book.pnl(  'heston', vol=vol_initial, params=heston_params):.4f}")
    print()
    print(book.summary('heston', vol=vol_initial, params=heston_params).to_string())
    print()

    # ── Mise à jour du spot et re-pricing ────────────────────────
    print("=== Après un choc spot à 95 ===")
    book.update_spot(95.0)
    print(f"MtM    : {book.price('blackScholes', vol=vol, params=bs_params):.4f}")
    print(f"Greeks : {book.greeks(vol=vol)}")
