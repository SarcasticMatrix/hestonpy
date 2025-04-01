:sd_hide_title:
:html_theme.sidebar_secondary.remove:

.. raw:: html

    <!-- CSS overrides on the homepage only -->
    <style>
    .bd-main .bd-content .bd-article-container {
    max-width: 70rem; /* Make homepage a little wider instead of 60em */
    }
    /* Extra top/bottom padding to the sections */
    article.bd-article section {
    padding: 3rem 0 7rem;
    }
    /* Override all h1 headers except for the hidden ones */
    h1:not(.sd-d-none) {
    font-weight: bold;
    font-size: 48px;
    text-align: center;
    margin-bottom: 4rem;
    }
    /* Override all h3 headers that are not in hero */
    h3:not(#hero h3) {
    font-weight: bold;
    text-align: center;
    }
    </style>

hestonpy
~~~~~~~~

.. grid:: 2

    .. grid-item-card::
        :shadow: none
        :class-card: no-border

        .. raw:: html

            <div id="hero-left">  <!-- Start Hero -->
            <h2 style="font-size: 60px; font-weight: bold; margin: 5rem auto 0;">hestonpy</h2>
            <h3 style="font-weight: bold; margin-top: 0;">Stochastic Volatility models</h3>
            <p>hestonpy is a Python package for the calibration of stochastic volatility models.</p>

            <div class="homepage-button-container">
            <div class="homepage-button-container-row">
                <a href="./getting_started/index.html" class="homepage-button primary-button">User guide</a>
                <a href="./examples/index.html" class="homepage-button secondary-button">Examples</a>
            </div>
            <div class="homepage-button-container-row">
                <a href="./api/index.html" class="homepage-button-link">See API Reference →</a>
            </div>
            </div>
            </div>  <!-- End Hero -->

    .. grid-item-card::
        :shadow: none
        :class-card: no-border

        .. image:: ./_static/_images/example.jpeg
            :width: 100%


Key features
~~~~~~~~~~~~

The ``hestonpy`` Python package implements the **Heston** and **Bates** models for option **pricing**, **hedging**, 
and robust **calibration** on implied volatility smiles. 
The package also includes functionality for optimal portfolio allocation using stochastic control techniques.

Covered topics by the ``hestonpy`` package:

* Heston, Bates, and Black-Scholes models (path simulations, options pricing, etc.).
* Calibration on implied volatility smile (from Yahoo Finance data, personal data, or synthetic data).
* De-noising market data (SVI implementation).
* Asset allocations (stochastic optimal control under Heston dynamics).

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   ./api/index.rst
   ./user_guide/index.rst
   ./examples/index.rst