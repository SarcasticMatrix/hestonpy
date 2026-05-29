~~~~~~~~~~~~
Explorations
~~~~~~~~~~~~

A collection of mathematical notes exploring questions that naturally arise when using ``hestonpy``.
Each post mixes theory, intuition, and concrete code to dig into one specific question.

.. grid:: 1
   :gutter: 4

   .. grid-item-card:: Is the implied volatility surface arbitrage-free?
      :link: arbitrage_free
      :link-type: doc
      :shadow: md

      We derive the conditions for a volatility surface to be free of **butterfly** and
      **calendar spread** arbitrage. Starting from the Durrleman condition, we show how
      SVI and SSVI parameterisations naturally handle — or violate — these constraints,
      and how to check them numerically with ``hestonpy``.

      +++
      **Topics:** SVI · SSVI · Durrleman condition · risk-neutral density


.. toctree::
   :hidden:
   :maxdepth: 1

   arbitrage_free
