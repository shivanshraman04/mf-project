# dashboards/factors_app.py
#
# Run with: streamlit run dashboards/factors_app.py
# Answers: what factors explain returns, at category and fund level?

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter

from src.config import AUM_PANEL_FILE, FACTOR_FILE
from src.analysis import load_factors

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Factor Dashboard',
    page_icon='🔬',
    layout='wide',
)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    panel   = pd.read_parquet(AUM_PANEL_FILE)
    factors = load_factors(FACTOR_FILE)
    panel   = panel[~panel['category'].str.contains('Hybrid', na=False)].copy()
    panel['cat_short'] = (panel['category']
        .str.replace('Equity Scheme - ', '', regex=False))
    panel['fund_short'] = (panel['scheme_name']
        .str.replace(' - Direct Plan - Growth', '', regex=False)
        .str.replace(' - Direct - Growth', '', regex=False)
        .str.replace(' Direct Plan Growth', '', regex=False))
    return panel, factors

panel, factors = load_data()

# ── helper functions ──────────────────────────────────────────────────────────
def stars(t):
    t = abs(t)
    if t >= 2.58: return '***'
    if t >= 1.96: return '**'
    if t >= 1.65: return '*'
    return ''


def aum_weighted_return(group):
    w     = group['aaum_lakhs_lagged']
    r     = group['monthly_return']
    valid = w.notna() & r.notna() & (w > 0)
    if valid.sum() == 0:
        return np.nan
    return np.average(r[valid], weights=w[valid])


@st.cache_data
def build_category_returns(panel_hash):
    panel_clean = panel.dropna(subset=['monthly_return']).copy()
    cat_aum = (panel_clean
        .groupby(['cat_short', 'year_month'])
        .apply(aum_weighted_return, include_groups=False)
        .unstack(level=0)
        .sort_index())
    return cat_aum


@st.cache_data
def run_category_regressions(panel_hash):
    cat_aum = build_category_returns(panel_hash)
    common  = cat_aum.index.intersection(factors.index)
    cat_f   = cat_aum.loc[common]
    fac     = factors.loc[common]

    loadings = {}
    for cat in cat_f.columns:
        y = cat_f[cat].dropna()
        if len(y) < 24:
            continue
        ye = y - fac.loc[y.index, 'RF']
        X  = sm.add_constant(fac.loc[y.index, ['MF', 'SMB', 'HML', 'WML']])
        m  = sm.OLS(ye, X).fit()
        loadings[cat] = {
            'alpha_ann':  m.params['const'] * 12,
            'alpha_t':    m.tvalues['const'],
            'beta_MF':    m.params['MF'],
            'beta_MF_t':  m.tvalues['MF'],
            'beta_SMB':   m.params['SMB'],
            'beta_SMB_t': m.tvalues['SMB'],
            'beta_HML':   m.params['HML'],
            'beta_HML_t': m.tvalues['HML'],
            'beta_WML':   m.params['WML'],
            'beta_WML_t': m.tvalues['WML'],
            'R2':         m.rsquared,
        }
    return pd.DataFrame(loadings).T.sort_values('beta_SMB', ascending=False)


panel_hash = str(panel.shape)
cat_aum    = build_category_returns(panel_hash)
loadings   = run_category_regressions(panel_hash)

# ── page layout ───────────────────────────────────────────────────────────────
st.title('🔬 Factor Dashboard')

tab1, tab2 = st.tabs(['Category Factor Loadings', 'Fund-Level Rolling Exposures'])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Category factor loadings
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('4-Factor Loadings by Category')
    st.caption('Greyed cells: |t| < 1.96 (not significant at 5%).  * p<10%  ** p<5%  *** p<1%')

    display_rows = []
    for cat, row in loadings.iterrows():
        display_rows.append({
            'Category': cat,
            'Alpha':    f"{row['alpha_ann']:+.2%}{stars(row['alpha_t'])}",
            'MF':       f"{row['beta_MF']:+.3f}{stars(row['beta_MF_t'])}",
            'SMB':      f"{row['beta_SMB']:+.3f}{stars(row['beta_SMB_t'])}",
            'HML':      f"{row['beta_HML']:+.3f}{stars(row['beta_HML_t'])}",
            'WML':      f"{row['beta_WML']:+.3f}{stars(row['beta_WML_t'])}",
            'R²':       f"{row['R2']:.3f}",
        })

    st.dataframe(
        pd.DataFrame(display_rows).set_index('Category'),
        use_container_width=True,
    )

    st.subheader('Factor Loadings Heatmap')

    plot_cols  = ['beta_MF', 'beta_SMB', 'beta_HML', 'beta_WML']
    tstat_cols = ['beta_MF_t', 'beta_SMB_t', 'beta_HML_t', 'beta_WML_t']
    plot_data  = loadings[plot_cols].astype(float)
    tstat_data = loadings[tstat_cols].astype(float).values
    insig_mask = np.abs(tstat_data) < 1.96

    annot = np.empty_like(plot_data.values, dtype=object)
    for i, cat in enumerate(plot_data.index):
        for j, (col, tcol) in enumerate(zip(plot_cols, tstat_cols)):
            coef        = plot_data.iloc[i, j]
            t           = loadings.loc[cat, tcol]
            annot[i, j] = f"{coef:+.2f}{stars(t)}"

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        plot_data,
        annot=annot, fmt='',
        cmap='RdBu_r', center=0,
        vmin=-1.5, vmax=1.5,
        linewidths=0.5,
        annot_kws={'size': 9},
        ax=ax,
    )

    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            if insig_mask[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=True, color='#e8e8e8',
                    alpha=0.75, zorder=3,
                ))
                ax.text(
                    j + 0.5, i + 0.5, annot[i, j],
                    ha='center', va='center',
                    fontsize=9, color='#888888', zorder=4,
                )

    ax.set_xticklabels(
        ['Market (MF)', 'Size (SMB)', 'Value (HML)', 'Momentum (WML)'],
        fontsize=10,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)
    ax.set_title(
        'Factor Loadings by Category\nGreyed = not significant at 5%',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader('Category Return Correlations')

    corr = cat_aum.corr()
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr.round(2),
        annot=True, fmt='.2f',
        cmap='RdYlGn_r',
        center=0, vmin=0.5, vmax=1.0,
        square=True,
        annot_kws={'size': 8},
        ax=ax2,
    )
    ax2.set_title('AUM-Weighted Return Correlations by Category',
                  fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Fund-level rolling factor exposures
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Fund-Level Rolling Factor Exposures')

    col_search, col_window = st.columns([3, 1])
    with col_search:
        fund_search = st.text_input(
            'Search fund by name',
            value='SBI Small Cap',
            placeholder='e.g. Mirae Asset Large Cap',
        )
    with col_window:
        roll_window = st.selectbox(
            'Rolling window (months)',
            options=[24, 36, 48],
            index=1,
        )

    if fund_search:
        match = panel[panel['fund_short'].str.contains(
            fund_search, case=False, na=False
        )]

        if len(match) == 0:
            st.warning(f"No fund found matching '{fund_search}'")
        else:
            fund_code = match['scheme_code'].iloc[0]
            fund_name = match['fund_short'].iloc[0]
            fund_cat  = match['cat_short'].iloc[0]

            st.markdown(f"**{fund_name}** — {fund_cat}")

            fund_data = panel[panel['scheme_code'] == fund_code].sort_values('year_month')

            fund_f = fund_data.merge(
                factors.reset_index()[['year_month', 'MF', 'SMB', 'HML', 'WML', 'RF']],
                on='year_month', how='inner'
            ).dropna(subset=['monthly_return', 'MF', 'SMB', 'HML', 'WML', 'RF'])

            if len(fund_f) < roll_window + 6:
                st.warning(
                    f"Not enough data for {roll_window}-month rolling window. "
                    f"Only {len(fund_f)} months available."
                )
            else:
                # Rolling 4-factor OLS
                roll_results = []
                for t in range(roll_window, len(fund_f)):
                    window = fund_f.iloc[t - roll_window:t]
                    ye     = window['monthly_return'] - window['RF']
                    X      = sm.add_constant(window[['MF', 'SMB', 'HML', 'WML']])
                    try:
                        m = sm.OLS(ye, X).fit()
                        roll_results.append({
                            'year_month':   fund_f.iloc[t]['year_month'],
                            'beta_MF':      m.params['MF'],
                            'beta_MF_t':    m.tvalues['MF'],
                            'beta_SMB':     m.params['SMB'],
                            'beta_SMB_t':   m.tvalues['SMB'],
                            'beta_HML':     m.params['HML'],
                            'beta_HML_t':   m.tvalues['HML'],
                            'beta_WML':     m.params['WML'],
                            'beta_WML_t':   m.tvalues['WML'],
                            'alpha_4f_ann': m.params['const'] * 12,
                            'alpha_4f_t':   m.tvalues['const'],
                            'capm_alpha':   np.nan,
                            'capm_alpha_t': np.nan,
                        })
                    except Exception:
                        continue

                # Rolling CAPM alpha
                for idx, t in enumerate(range(roll_window, len(fund_f))):
                    if idx >= len(roll_results):
                        break
                    window = fund_f.iloc[t - roll_window:t]
                    ye     = window['monthly_return'] - window['RF']
                    X      = sm.add_constant(window[['MF']])
                    try:
                        m = sm.OLS(ye, X).fit()
                        roll_results[idx]['capm_alpha']   = m.params['const'] * 12
                        roll_results[idx]['capm_alpha_t'] = m.tvalues['const']
                    except Exception:
                        pass

                roll_df       = pd.DataFrame(roll_results)
                roll_df['date'] = roll_df['year_month'].dt.to_timestamp()
                latest          = roll_df.iloc[-1]

                # ── summary metrics ───────────────────────────────────────────
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric('Market Beta',
                          f"{latest['beta_MF']:+.3f}",
                          f"t = {latest['beta_MF_t']:+.2f}")
                c2.metric('SMB Beta',
                          f"{latest['beta_SMB']:+.3f}",
                          f"t = {latest['beta_SMB_t']:+.2f}")
                c3.metric('HML Beta',
                          f"{latest['beta_HML']:+.3f}",
                          f"t = {latest['beta_HML_t']:+.2f}")
                c4.metric('WML Beta',
                          f"{latest['beta_WML']:+.3f}",
                          f"t = {latest['beta_WML_t']:+.2f}")
                c5.metric('4F Alpha (ann.)',
                          f"{latest['alpha_4f_ann']:+.2%}",
                          f"t = {latest['alpha_4f_t']:+.2f}")

                st.caption(
                    f"Latest values from {roll_window}-month window "
                    f"ending {roll_df['year_month'].iloc[-1]}"
                )

                # ── rolling betas chart ───────────────────────────────────────
                fig3, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
                axes = axes.flatten()

                beta_specs = [
                    ('beta_MF',  'beta_MF_t',  'Market Beta (MF)',   '#1f4e79'),
                    ('beta_SMB', 'beta_SMB_t', 'Size Beta (SMB)',    '#c0392b'),
                    ('beta_HML', 'beta_HML_t', 'Value Beta (HML)',   '#27ae60'),
                    ('beta_WML', 'beta_WML_t', 'Momentum Beta (WML)','#8e44ad'),
                ]

                for ax, (col, tcol, label, color) in zip(axes, beta_specs):
                    coefs = roll_df[col].values
                    tstats = roll_df[tcol].values
                    dates  = roll_df['date'].values

                    ax.plot(dates, coefs, color=color, linewidth=1.8)
                    ax.axhline(0, color='gray', linestyle='--',
                               linewidth=0.8, alpha=0.5)

                    # Shade regions where |t| < 1.96 (not significant)
                    insig = np.abs(tstats) < 1.96
                    ax.fill_between(
                        dates, coefs, 0,
                        where=insig,
                        alpha=0.15, color=color,
                        label='Not significant (|t|<1.96)',
                    )

                    ax.set_title(label, fontsize=10, fontweight='bold')
                    ax.legend(fontsize=7, frameon=False)
                    ax.grid(True, alpha=0.25)
                    for sp in ['top', 'right']:
                        ax.spines[sp].set_visible(False)

                fig3.suptitle(
                    f'{fund_name[:55]} — {roll_window}-Month Rolling Factor Exposures',
                    fontsize=11, fontweight='bold', y=1.01,
                )
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

                # ── rolling alpha chart ───────────────────────────────────────
                st.subheader('Rolling Alpha')

                fig4, ax4 = plt.subplots(figsize=(14, 4))
                ax4.plot(roll_df['date'], roll_df['capm_alpha'],
                         color='#e67e22', linewidth=1.8, label='CAPM Alpha')
                ax4.plot(roll_df['date'], roll_df['alpha_4f_ann'],
                         color='#2c3e50', linewidth=1.8,
                         linestyle='--', label='4-Factor Alpha')
                ax4.axhline(0, color='gray', linestyle='-', linewidth=0.8)
                ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
                ax4.set_ylabel('Annualised Alpha')
                ax4.legend(fontsize=10, frameon=False)
                ax4.set_title(
                    f'Rolling CAPM vs 4-Factor Alpha ({roll_window}-month window)',
                    fontsize=10, fontweight='bold',
                )
                ax4.grid(True, alpha=0.25)
                for sp in ['top', 'right']:
                    ax4.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()

                # ── raw data expander ─────────────────────────────────────────
                with st.expander('Show raw rolling estimates'):
                    display_roll = roll_df[[
                        'year_month',
                        'beta_MF',  'beta_MF_t',
                        'beta_SMB', 'beta_SMB_t',
                        'beta_HML', 'beta_HML_t',
                        'beta_WML', 'beta_WML_t',
                        'capm_alpha', 'capm_alpha_t',
                        'alpha_4f_ann', 'alpha_4f_t',
                    ]].copy()

                    display_roll.columns = [
                        'Month',
                        'MF Beta', 'MF t',
                        'SMB Beta', 'SMB t',
                        'HML Beta', 'HML t',
                        'WML Beta', 'WML t',
                        'CAPM Alpha', 'CAPM t',
                        '4F Alpha', '4F t',
                    ]

                    for col in ['CAPM Alpha', '4F Alpha']:
                        display_roll[col] = display_roll[col].map('{:+.2%}'.format)
                    for col in ['MF Beta', 'SMB Beta', 'HML Beta', 'WML Beta']:
                        display_roll[col] = display_roll[col].map('{:+.3f}'.format)
                    for col in ['MF t', 'SMB t', 'HML t', 'WML t',
                                'CAPM t', '4F t']:
                        display_roll[col] = display_roll[col].map('{:+.2f}'.format)

                    st.dataframe(display_roll, use_container_width=True)