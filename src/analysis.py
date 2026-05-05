# src/analysis.py
#
# Core quantitative functions for the mf-allocation project.
# Factor loading, rolling alpha computation, decile analysis, regressions.

import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.config import (
    FACTOR_DATA_END,
    ALPHA_LOOKBACK_MONTHS,
    MIN_OBSERVATIONS,
    N_DECILES,
    MIN_FUNDS_PER_DECILE,
    RISK_FREE_ANNUAL,
)


def load_factors(filepath):
    """
    Load and clean the IIMA 4-factor CSV.

    The CSV stores factor returns as percentages (e.g. 1.5 means 1.5%).
    We divide by 100 to convert to decimals for use in OLS regressions.
    Clips to FACTOR_DATA_END so analysis never silently runs beyond
    available factor data.

    Returns a DataFrame indexed by year_month (pd.Period), columns:
        MF, SMB, HML, WML, RF
    """
    factors = pd.read_csv(filepath)

    factors['year_month'] = (
        pd.to_datetime(factors['Date'].astype(str), format='%Y-%m')
        .dt.to_period('M')
    )
    factors = factors.drop(columns=['Date'])
    factors = factors.replace('NA', np.nan)

    for col in ['SMB', 'HML', 'WML', 'MF', 'RF']:
        factors[col] = pd.to_numeric(factors[col], errors='coerce')

    # Convert from percentage to decimal
    factors[['SMB', 'HML', 'WML', 'MF', 'RF']] /= 100

    # Clip to available factor data
    factors = factors[
        factors['year_month'] <= pd.Period(FACTOR_DATA_END, freq='M')
    ]

    return factors.set_index('year_month').sort_index()


def summary_stats(r, rf_annual=RISK_FREE_ANNUAL):
    """
    Compute annualised performance statistics for a monthly return series.

    Args:
        r:         pd.Series of monthly returns in decimal form
        rf_annual: annual risk-free rate for Sharpe ratio calculation

    Returns a pd.Series with:
        CAGR, Vol, Sharpe, MaxDD, N
    """
    r = r.dropna()
    if len(r) < 12:
        return pd.Series({
            'CAGR': np.nan, 'Vol': np.nan,
            'Sharpe': np.nan, 'MaxDD': np.nan, 'N': len(r)
        })

    rf_monthly = rf_annual / 12
    cagr       = (1 + r).prod() ** (12 / len(r)) - 1
    vol        = r.std() * np.sqrt(12)
    sharpe     = (r - rf_monthly).mean() / r.std() * np.sqrt(12)
    cum        = (1 + r).cumprod()
    maxdd      = ((cum / cum.cummax()) - 1).min()

    return pd.Series({
        'CAGR': cagr, 'Vol': vol,
        'Sharpe': sharpe, 'MaxDD': maxdd, 'N': len(r)
    })


def compute_trailing_alpha(panel, factors,
                           lookback=ALPHA_LOOKBACK_MONTHS,
                           min_obs=MIN_OBSERVATIONS):
    """
    Compute rolling 4-factor alpha for each fund-month.

    For each fund, at each month t, runs OLS on the prior `lookback` months:
        excess_return = alpha + b1*MF + b2*SMB + b3*HML + b4*WML + e

    Alpha is stored in monthly decimal form (multiply by 12 to annualise).

    This is the slow function — expect 5-10 minutes for 618 funds.
    Run once and save output to ALPHA_FILE. Do not re-run unless the
    panel or factors have been updated.

    Args:
        panel:   monthly returns panel with columns:
                 scheme_code, year_month, monthly_return
        factors: DataFrame from load_factors(), indexed by year_month
        lookback: rolling window in months (default from config)
        min_obs:  minimum valid observations required (default from config)

    Returns a DataFrame with columns:
        scheme_code, year_month, trailing_alpha, trailing_alpha_t
    """
    panel = panel.sort_values(['scheme_code', 'year_month']).reset_index(drop=True)

    # Merge factors onto panel
    panel_f = panel.merge(
        factors.reset_index()[['year_month', 'MF', 'SMB', 'HML', 'WML', 'RF']],
        on='year_month',
        how='left'
    )
    panel_f['excess_return'] = panel_f['monthly_return'] - panel_f['RF']

    alphas = []
    funds  = panel_f['scheme_code'].unique()
    n      = len(funds)

    for i, code in enumerate(funds):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} funds processed")

        fund = (panel_f[panel_f['scheme_code'] == code]
                .sort_values('year_month')
                .reset_index(drop=True))

        for t in range(lookback, len(fund)):
            window = fund.iloc[t - lookback:t]
            window = window.dropna(
                subset=['excess_return', 'MF', 'SMB', 'HML', 'WML']
            )

            if len(window) < min_obs:
                continue

            X = sm.add_constant(window[['MF', 'SMB', 'HML', 'WML']])
            y = window['excess_return']

            try:
                m = sm.OLS(y, X).fit()
                alphas.append({
                    'scheme_code':      code,
                    'year_month':       fund.iloc[t]['year_month'],
                    'trailing_alpha':   m.params['const'],
                    'trailing_alpha_t': m.tvalues['const'],
                })
            except Exception:
                continue

    return pd.DataFrame(alphas)


def build_decile_panel(panel, factors, trailing_alphas=None,
                       ranking_signal='return',
                       ranking_months=12,
                       holding_months=12,
                       n_deciles=N_DECILES,
                       min_funds_per_decile=MIN_FUNDS_PER_DECILE,
                       min_months_in_ranking_year=10):
    """
    Core persistence test engine.

    At each rebalance date:
        1. Compute or load the ranking signal for each fund
        2. Filter funds with insufficient history
        3. Sort into deciles — D1 = best signal, D{n} = worst
        4. Hold fixed composition for holding_months
        5. Compute equal-weighted monthly return per decile per month

    Rebalances annually at December of each available year.
    Within a holding period the decile composition is fixed —
    funds do not move between deciles until the next rebalance.

    Args:
        panel:            monthly returns panel (scheme_code, year_month,
                          monthly_return, aaum_lakhs_lagged)
        factors:          from load_factors() — used only for alpha signal
        trailing_alphas:  from compute_trailing_alpha() — required if
                          ranking_signal == 'alpha', else ignored
        ranking_signal:   'return' or 'alpha'
        ranking_months:   months of history used to compute return signal
                          (ignored if ranking_signal == 'alpha')
        holding_months:   months to hold after each rebalance date
        n_deciles:        number of deciles (default from config)
        min_funds_per_decile: skip rebalance year if any decile has fewer
                          funds than this (default from config)
        min_months_in_ranking_year: minimum valid months a fund needs in
                          the ranking window to be eligible

    Returns:
        decile_wide: DataFrame indexed by year_month, columns D1..D{n} 
                     and Spread (D1 - D{n}), containing monthly returns
        meta:        dict with coverage info
    """
    if ranking_signal == 'alpha' and trailing_alphas is None:
        raise ValueError("trailing_alphas must be provided when ranking_signal='alpha'")

    panel = panel.sort_values(['scheme_code', 'year_month']).reset_index(drop=True)
    all_months = sorted(panel['year_month'].unique())

    # Rebalance at December of each year that exists in the panel
    rebalance_dates = [
        pd.Period(f'{m.year}-12', freq='M')
        for m in all_months
        if m.month == 12
    ]

    decile_returns  = []
    valid_rebalances = 0

    for rb_date in rebalance_dates:

        # ── compute ranking signal ────────────────────────────────────────
        if ranking_signal == 'return':
            # Cumulative return over prior ranking_months ending at rb_date
            window_start = rb_date - ranking_months
            rank_window  = panel[
                (panel['year_month'] >  window_start) &
                (panel['year_month'] <= rb_date)
            ]
            fund_signal = (
                rank_window
                .groupby('scheme_code')
                .apply(lambda g: (
                    (1 + g['monthly_return'].dropna()).prod() - 1
                    if g['monthly_return'].notna().sum() >= min_months_in_ranking_year
                    else np.nan
                ))
                .dropna()
            )

        else:  # alpha
            # Use trailing alpha at the rebalance date
            rank_data   = trailing_alphas[
                trailing_alphas['year_month'] == rb_date
            ]
            fund_signal = (
                rank_data.set_index('scheme_code')['trailing_alpha']
                .dropna()
            )

        if len(fund_signal) < n_deciles * min_funds_per_decile:
            print(f"  {rb_date}: insufficient funds ({len(fund_signal)}), skipping")
            continue

        # ── assign deciles ────────────────────────────────────────────────
        # rank(method='first') breaks ties by order of appearance
        # D1 = highest signal, D{n} = lowest
        decile_labels = pd.qcut(
            fund_signal.rank(method='first'),
            n_deciles,
            labels=range(1, n_deciles + 1)
        )
        decile_assign = pd.DataFrame({
            'scheme_code': fund_signal.index,
            'decile':      decile_labels.values
        })

        # Skip if any decile is too thin
        decile_counts = decile_assign['decile'].value_counts()
        if decile_counts.min() < min_funds_per_decile:
            print(f"  {rb_date}: thin decile ({decile_counts.min()} funds), skipping")
            continue

        # ── holding period ────────────────────────────────────────────────
        hold_start = rb_date + 1
        hold_end   = rb_date + holding_months

        hold_data = panel[
            (panel['year_month'] >= hold_start) &
            (panel['year_month'] <= hold_end)
        ].merge(decile_assign, on='scheme_code', how='inner')

        if len(hold_data) == 0:
            continue

        # Equal-weighted return per decile per month
        # Composition is fixed — same funds throughout the holding period
        dm = (
            hold_data
            .groupby(['decile', 'year_month'], observed=True)['monthly_return']
            .mean()
            .reset_index()
        )
        dm['rebalance'] = str(rb_date)
        decile_returns.append(dm)
        valid_rebalances += 1

    if not decile_returns:
        raise ValueError("No valid rebalance periods found. Check panel coverage.")

    # ── build wide format ─────────────────────────────────────────────────
    dp = pd.concat(decile_returns, ignore_index=True)
    decile_wide = (
        dp.pivot_table(
            index='year_month', columns='decile',
            values='monthly_return', aggfunc='mean'
        )
        .sort_index()
    )
    decile_wide.columns = [f'D{int(c)}' for c in decile_wide.columns]

    # Spread: long D1 (best prior signal), short D{n} (worst prior signal)
    # Positive spread = persistence. Negative spread = mean reversion.
    decile_wide['Spread'] = decile_wide['D1'] - decile_wide[f'D{n_deciles}']

    meta = {
        'n_months':      len(decile_wide),
        'start':         decile_wide.index.min(),
        'end':           decile_wide.index.max(),
        'n_rebalances':  valid_rebalances,
        'ranking_signal': ranking_signal,
        'ranking_months': ranking_months,
        'holding_months': holding_months,
    }

    return decile_wide, meta


def run_regressions(decile_wide, factors):
    """
    Run 4-factor OLS on each decile portfolio and the spread.

    The spread portfolio is regressed directly as a single time series —
    this is the correct way to compute the spread t-statistic because it
    accounts for the covariance between D1 and D{n}. Subtracting two
    separately estimated t-stats would not be valid.

    Args:
        decile_wide: from build_decile_panel()
        factors:     from load_factors()

    Returns a DataFrame indexed by decile (D1..D{n}, Spread) with columns:
        CAGR, Sharpe, ff4_alpha, ff4_alpha_t, beta_MF, beta_SMB,
        beta_HML, beta_WML, R2
    """
    common = decile_wide.index.intersection(factors.index)
    if len(common) < 12:
        raise ValueError(f"Only {len(common)} common months — insufficient for regression")

    dec = decile_wide.loc[common]
    fac = factors.loc[common]

    results = {}

    for col in dec.columns:
        y = dec[col].dropna()
        if len(y) < 12:
            continue

        ye = y - fac.loc[y.index, 'RF']
        X  = sm.add_constant(fac.loc[y.index, ['MF', 'SMB', 'HML', 'WML']])
        m  = sm.OLS(ye, X).fit()

        cagr   = (1 + y).prod() ** (12 / len(y)) - 1
        sharpe = (ye.mean() / y.std()) * np.sqrt(12)

        results[col] = {
            'CAGR':        cagr,
            'Sharpe':      sharpe,
            'ff4_alpha':   m.params['const'] * 12,
            'ff4_alpha_t': m.tvalues['const'],
            'beta_MF':     m.params['MF'],
            'beta_SMB':    m.params['SMB'],
            'beta_HML':    m.params['HML'],
            'beta_WML':    m.params['WML'],
            'R2':          m.rsquared,
        }

    return pd.DataFrame(results).T  # outside the for loop
