:sd_hide_title:
:html_theme.sidebar_secondary.remove:

.. raw:: html

    <!-- CSS overrides on the homepage only -->
    <style>
    .bd-main .bd-content .bd-article-container {
    max-width: 75rem; /* Make homepage a little wider instead of 60em */
    }
    /* Extra top/bottom padding to the sections */
    article.bd-article section {
    padding: 7rem 0 7rem;
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

hestonpy: Calibration of stochastic volatility models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. raw:: html

    <div style="display: flex;">

        <div style="flex: 1;">
            <div id="hero-left">  <!-- Start Hero -->
                <h2 style="font-size: 60px; font-weight: bold; margin: 5rem auto 0;">hestonpy</h2>
                <h3 style="font-size: 25px; font-weight: bold; margin-top: 0;">Stochastic Volatility models</h3>
                <p>
                    <code>hestonpy</code> implements <b>Heston</b> and <b>Bates</b> models for options <b>pricing</b>, 
                    <b>hedging</b>, and robust <b>calibration</b> on implied volatility smiles.
                </p>

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
        </div>

        <div style="flex: 2; display: flex; flex-direction: column;">

            <div style="flex: 1;">
                <img src="_static/_images/smile-bates.png" alt="implied vol market smile" style="width: 100%;">
            </div>

            <div style="flex: 1;">

            </div>
        </div>

    </div>



Key features
~~~~~~~~~~~~

**It is an user friendly, intuitive and humble python library. Please, contact me if you have any suggestions! :)**


The ``hestonpy`` Python package implements the **Heston** and **Bates** models for option **pricing**, **hedging**, 
and robust **calibration** on implied volatility smiles. 
The package also includes functionality for optimal portfolio allocation using stochastic control techniques.

Covered topics by the ``hestonpy`` package:

* Heston, Bates, and Black-Scholes models (vanilla options pricing, hedging).
* Calibration on implied volatility smile (from Yahoo Finance data, personal data, or synthetic data).
* De-noising market data (SVI implementation and filters).
* Asset allocations (stochastic optimal control under Heston dynamics).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./api/index.rst
   ./user_guide/index.rst
   ./examples/index.rst