import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm, jarque_bera
from scipy.signal import periodogram


def residuals_analysis(residuals: np.array, quantileExtremValue: int = 5):

    ##############################################################################
    # Statististics
    ##############################################################################
    print(" \n Some Statistics ...")

    print(
        "Biais:",
        round(np.mean(residuals), 3),
        " - Standard deviation:",
        round(np.std(residuals), 3),
    )
    ##############################################################################
    # Statistical tests
    ##############################################################################
    print(" \n Some Statistical tests ... \n")

    # If the p-value is less than the significance level (0.05),
    # reject the null hypothesis of Ljung-Box test of autocorrelation in residuals.
    pvalue = round(
        sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)[
            "lb_pvalue"
        ].values[0],
        3,
    )
    if pvalue < 0.05:
        print(f"Autocorrelation in residuals detected - pvalue: {pvalue}")
    else:
        print(f"No autocorrelation in residuals detected - pvalue: {pvalue}")

    # If the p-value is less than the chosen significance level (0.05),
    # reject the null hypothesis of Jarque-Bera test of skewness and kurtosis unmatching a normal distribution.
    pvalue = round(jarque_bera(residuals).pvalue, 3)
    if pvalue < 0.05:
        print(
            f"Residuals Skewness and Kurtosis unmatched with a normal distribution residuals detected - pvalue: {pvalue}"
        )
    else:
        print(
            f"Residuals Skewness and Kurtosis matched with a normal distribution residuals detected - pvalue: {pvalue}"
        )

    ##############################################################################
    # Plots
    ##############################################################################

    # Set up subplots
    fig, axes = plt.subplots(nrows=2, ncols=3)

    mean_resid = np.mean(residuals)
    sd_resid = np.std(residuals)
    x = np.linspace(min(residuals), max(residuals), 300)

    # ----------------------------------------------------------------------------
    # ACF
    # ----------------------------------------------------------------------------
    plot_acf(residuals, lags=6 * 12, ax=axes[0, 2], marker=None, linewidth=0.5)
    axes[0, 2].set_title("ACF")

    # ----------------------------------------------------------------------------
    # Histogram
    # ----------------------------------------------------------------------------
    sns.histplot(
        residuals,
        kde=True,
        stat="density",
        ax=axes[1, 1],
        label="Density of Residuals",
        linewidth=1.5,
    )
    axes[1, 1].set_title("Histogram")
    axes[1, 1].plot(
        x,
        norm.pdf(x, mean_resid, sd_resid),
        color="red",
        linewidth=1.5,
        label="Normal Density",
    )
    axes[1, 1].legend(loc="upper right")

    # ----------------------------------------------------------------------------
    # Raw periodogram
    # ----------------------------------------------------------------------------
    f, Pxx_den = periodogram(residuals, 10e3)
    axes[0, 0].semilogy(f[1:] / (np.max(f) * 2), Pxx_den[1:], linewidth=0.5)
    axes[0, 0].set_xlabel("Normalized frequency")
    axes[0, 0].set_ylabel("Power spectral density")
    axes[0, 0].set_title("Raw periodogram")

    # ----------------------------------------------------------------------------
    # Cumulated periodogram
    # ----------------------------------------------------------------------------
    cumulative_periodogram = np.cumsum(Pxx_den)
    axes[0, 1].plot(f / np.max(f), cumulative_periodogram, color="blue", linewidth=1)
    axes[0, 1].plot(
        [0, max(f / np.max(f))],
        [0, max(cumulative_periodogram)],
        color="red",
        linewidth=1,
        label="White noise",
    )
    axes[0, 1].set_xlabel("Cumulated Normalized frequency")
    axes[0, 1].set_ylabel("Cumulated Power spectral density")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_title("Cumulated periodogram")

    # ----------------------------------------------------------------------------
    # Quantile-quantile plot
    # ----------------------------------------------------------------------------
    sm.qqplot(residuals, ax=axes[1, 0], linewidth=0.5, markersize=2, line="q")
    axes[1, 0].set_title("Quantile-quantile plot")

    # ----------------------------------------------------------------------------
    # Tail plot
    # ----------------------------------------------------------------------------
    my_kde = sns.kdeplot(
        residuals, ax=axes[1, 2], label="Residuals Density", linewidth=0.5, color="blue"
    )

    # create the upper limit
    x, y = my_kde.lines[0].get_data()
    quantile = np.percentile(residuals, quantileExtremValue)
    upper_limit = np.max(y[x <= quantile])

    axes[1, 2].plot(
        x,
        norm.pdf(x, mean_resid, sd_resid),
        color="red",
        linewidth=1,
        label="Normal Density",
    )
    axes[1, 2].set_ylim([0, upper_limit/3])
    axes[1, 2].legend()
    axes[1, 2].set_title("Tail of the Distribution")

    plt.show()
