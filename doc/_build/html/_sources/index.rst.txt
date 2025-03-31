.. hestonpy documentation master file, created by
   sphinx-quickstart on Mon Mar 31 09:01:06 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hestonpy documentation
======================


The ``hestonpy`` Python package implements the **Heston** and **Baites** models for option **pricing**, **hedging** 
and robust **calibration** on implied volatilitiy smiles. 
The package also includes functionality for optimal portfolio allocation using stochastic control techniques.

Covered topics by the ``hestonpy`` package:

* Heston, Bates and BlackScholes models (path simulations, options pricing etc.).
* Calibration on implied volatility smile (from yahoo finance data, personnal data or syntethic data).
* De-noising market data (SVI implementation).
* Asset allocations (stochastic optimal control under Heston dynamics).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./pages/models.rst

   ./pages/calibration.rst
