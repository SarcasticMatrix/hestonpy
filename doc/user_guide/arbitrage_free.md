# Is the implied volatility surface arbitrage-free?

```{admonition} Topics
:class: seealso

SVI · SSVI · Durrleman condition · risk-neutral density · butterfly arbitrage · calendar spread
```

When we calibrate an implied volatility surface from market quotes, we get a smooth,
good-looking surface. But does it *make sense* mathematically? Concretely: could a trader
build a portfolio of vanilla options from this surface and lock in a **guaranteed profit at
zero cost**? If yes, the surface is *not* arbitrage-free, it is internally inconsistent,
regardless of how well it fits the data.

This note works through two classical notions of arbitrage on a vol surface, derives the
conditions for each, and shows how to check them on a surface calibrated with `hestonpy`.

---

## Setup: total implied variance

Let $F_T = S_0 e^{r T}$ denote the forward price at maturity $T$ and spot $S_0$, and let $k = \ln(K / F_T)$ denote the log-moneyness. Instead of working with implied volatility $\sigma_{imp}(k,T)$ directly,
it is more convenient to work with the total implied variance:

$$
w(k, T) \;=\; T\sigma_{imp}(k, T)^2.
$$

This removes the trivial $\sqrt{T}$ scaling and makes the arbitrage conditions cleaner.

---

## Two types of arbitrage

### Calendar spread arbitrage

A **calendar spread** is a position long one call at maturity $T_2$ and short one call at
maturity $T_1 < T_2$, at the same strike. Under any risk-neutral measure, holding an option
to $T_2$ is worth at least as much as holding it to $T_1$ (early exercise aside), so the
forward calendar spread price must be non-negative.

Translating this into the language of total variance, The total variance must be non-decreasing in maturity at every strike. Intuitively: more time means more uncertainty, so a wider distribution of terminal prices.
Thus, one can show that a vol surface is free of calendar spread arbitrage if and only if, for every log-moneyness $k$:

$$
\partial_T\, w(k, T) \;\geq\; 0.
$$

### Butterfly arbitrage and the Durrleman condition

A **butterfly** centered at strike $K_0$ is long calls at $K_0 - \delta$ and $K_0 + \delta$
and short two calls at $K_0$. Its payoff is always non-negative (it's a tent function), so
its price must be non-negative too. A violation implies a negative risk-neutral density,
the model assigns negative probability to some region of the spot, which is absurd.

For a single maturity slice $w(\cdot) = w(\cdot, T)$, define Durrleman's function:

$$
g(k) \;=\; \left(1 - \frac{k\,w'(k)}{2\,w(k)}\right)^2
        \;-\; \frac{w'(k)^2}{4}\!\left(\frac{1}{w(k)} + \frac{1}{4}\right)
        \;+\; \frac{w''(k)}{2},
$$

where primes denote derivatives with respect to $k$. One can show that the risk-neutral
density of the log-moneyness $k$ satisfies:

$$
p(k) \;=\; \frac{e^{-d_2(k)^2/2}}{\sqrt{2\pi\, w(k)}} \cdot g(k), \qquad
d_2(k) = -\frac{k}{\sqrt{w(k)}} - \frac{\sqrt{w(k)}}{2}.
$$

Since $e^{-d_2^2/2} / \sqrt{2\pi w} > 0$ always, the density is non-negative if and only
if $g(k) \geq 0$. Therefore:

$$
\text{No butterfly arbitrage} \iff g(k) \geq 0 \quad \forall\, k.
$$

---

## Conditions for SVI

The **raw SVI** parameterisation (Gatheral, 2004) reads:

$$
w^{SVI}(k) = a + b\!\left(\rho(k - m) + \sqrt{(k-m)^2 + \sigma^2}\right).
$$

`hestonpy` calibrates this with the constraint $a + b\sigma\sqrt{1-\rho^2} \geq 0$, which
ensures the minimum total variance is non-negative. But is this *sufficient* for no
butterfly arbitrage?

No, in general. The constraint is **necessary** (negative total variance immediately implies
$g < 0$) but not sufficient. The full butterfly-free characterisation for SVI requires
checking $g(k) \geq 0$ everywhere. However, in practice a well-calibrated SVI slice
almost never violates it. The dangerous regime is when the wings are too steep
($b$ large, $\sigma$ small).

```{admonition} Necessary conditions for SVI (no butterfly)
:class: tip

- $b \geq 0$
- $|\rho| < 1$
- $a + b\sigma\sqrt{1-\rho^2} \geq 0$

These are enforced automatically by the `StochasticVolatilityInspired.calibration()` method.
```

### Checking the Durrleman condition numerically

For an SVI slice, $g(k)$ can be evaluated analytically by computing $w'$ and $w''$:

$$
w'(k) = b\left(\rho + \frac{k - m}{\sqrt{(k-m)^2 + \sigma^2}}\right),
$$

$$
w''(k) = b \cdot \frac{\sigma^2}{\left((k-m)^2 + \sigma^2\right)^{3/2}}.
$$

Note that $w'' > 0$ always (SVI is convex in $k$), which is a good sign. We plug these
into $g(k)$ and check the sign over a grid of log-moneyness values.

```python
import numpy as np
from hestonpy.models.calibration.svi import StochasticVolatilityInspired

# --- Calibrate SVI on a single smile ---
T = 0.5
svi = StochasticVolatilityInspired(time_to_maturity=T)

# (assume strikes, market_ivs, forward are available)
params, model_ivs = svi.calibration(strikes, market_ivs, forward)
a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]

# --- Durrleman condition on a fine grid ---
k_grid = np.linspace(-0.5, 0.5, 500)

sqrt_term  = np.sqrt((k_grid - m)**2 + sigma**2)
w   =  a + b * (rho * (k_grid - m) + sqrt_term)
w_p =  b * (rho + (k_grid - m) / sqrt_term)
w_pp = b * sigma**2 / sqrt_term**3

g = (1 - k_grid * w_p / (2 * w))**2 \
    - (w_p**2 / 4) * (1/w + 0.25)   \
    + w_pp / 2

is_butterfly_free = np.all(g >= 0)
print(f"Butterfly-free: {is_butterfly_free}")
print(f"Min g(k) = {g.min():.6f}")
```

---

## Conditions for SSVI

The **SSVI** parameterisation (Gatheral & Jacquier, 2014) writes the total variance surface as:

$$
w^{SSVI}(k, T) = \frac{\theta_T}{2}\!\left(
    1 + \rho_T\,\varphi_T\,k
    + \sqrt{(\varphi_T k + \rho_T)^2 + 1 - \rho_T^2}
\right),
$$

where $\theta_T = \sigma_{ATM}(T)^2 \cdot T$ is the ATM total variance and $\varphi_T \geq 0$
is the *wing* (curvature) parameter.

The strength of SSVI is that Gatheral & Jacquier derived **analytical no-arbitrage conditions**:

```{admonition} No-arbitrage theorem for SSVI (Gatheral & Jacquier, 2014)
:class: important

**No butterfly arbitrage** iff, for every maturity $T$:

$$\theta_T\,\varphi_T\,(1 + |\rho_T|) \leq 4.$$

**No calendar spread arbitrage** iff, for every pair of maturities $T_1 < T_2$ and every $k$:

$$w(k, T_1) \leq w(k, T_2).$$

In practice, since $\theta_T = \sigma_{ATM}^2\cdot T$ is set directly from market ATM vols,
calendar-spread arbitrage reduces to checking that the total ATM variance
$T \mapsto \theta_T$ is non-decreasing — which the market almost always satisfies.
```

In `hestonpy`, the calibration of each slice already enforces $\varphi_T \leq 2/\theta_T$,
which implies $\theta_T \varphi_T \leq 2$. Combined with $(1 + |\rho|) \leq 2$, this gives
$\theta_T \varphi_T (1 + |\rho_T|) \leq 4$ — exactly the no-butterfly condition.

```{admonition} Key insight
:class: note

The bound constraint `phi <= 2/theta_T` in the `calibrate_single_maturity` optimisation
is not just a numerical trick — it is a direct encoding of the no-butterfly condition
(up to the $\rho$ factor, which is bounded by $2$).
```

### Checking analytically with hestonpy

```python
import numpy as np
from hestonpy.models.calibration.ssvi import SurfaceStochasticVolatilityInspired

maturities = np.array([0.25, 0.5, 1.0, 2.0])
ssvi = SurfaceStochasticVolatilityInspired(maturities=maturities)

# (assume strikes, iv_surface, forwards are available)
params = ssvi.calibrate_surface(strikes, iv_surface, forwards)

print("Maturity | theta   | phi     | rho     | theta*phi*(1+|rho|) | OK?")
print("-" * 70)
for T in maturities:
    theta = ssvi.theta[T]
    rho   = params[T]["rho"]
    phi   = params[T]["phi"]
    cond  = theta * phi * (1 + abs(rho))
    ok    = "✓" if cond <= 4 else "✗ ARBITRAGE"
    print(f"  T={T:.2f}  | {theta:.4f} | {phi:.4f} | {rho:+.4f} | {cond:.4f}              | {ok}")
```

### Calendar spread check

```python
# Check that total ATM variance is non-decreasing in T
thetas = np.array([ssvi.theta[T] for T in maturities])
diffs  = np.diff(thetas)

if np.all(diffs >= 0):
    print("✓ No calendar spread arbitrage (theta is non-decreasing)")
else:
    bad = np.where(diffs < 0)[0]
    for i in bad:
        print(f"✗ Calendar spread arbitrage between T={maturities[i]:.2f} and T={maturities[i+1]:.2f}")
```

---

## Discussion

| | **SVI** | **SSVI** |
|---|---|---|
| Butterfly-free | Check $g(k) \geq 0$ numerically | Analytical: $\theta\varphi(1+\vert\rho\vert)\leq 4$ |
| Calendar-spread-free | Must compare slices | $\theta_T$ non-decreasing |
| Enforced in `hestonpy` | Necessary conditions only | ✓ via bound on $\varphi$ |

The practical takeaway: **SSVI is strictly safer than slice-by-slice SVI** for building a
full surface, because its no-arbitrage conditions can be encoded as simple box constraints in
the optimiser. SVI is more flexible on a single smile but offers no cross-maturity guarantee.

For a production calibration pipeline, one should:

1. Use SSVI (or parametric SSVI with $\varphi(\theta) = \eta/\theta^\gamma$) for the full
   surface, with the bound $\varphi \leq 2/\theta$.
2. After calibration, run a quick check: compute $g(k)$ on a fine grid for each slice.
3. Verify that $T \mapsto \theta_T$ is non-decreasing.

If any check fails, tighten the optimisation constraints or review the input data for
outlier quotes.

---

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization
  with application to the valuation of volatility derivatives.* Presentation at Global
  Derivatives & Risk Management, Madrid.
- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.*
  Quantitative Finance, 14(1), 59–71.
- Durrleman, V. (2005). *From implied to spot volatilities.* PhD thesis, Princeton University.
- Lee, R. (2004). *The moment formula for implied volatility at extreme strikes.*
  Mathematical Finance, 14(3), 469–480.
