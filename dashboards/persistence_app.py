# dashboards/persistence_app.py
#
# Run with: streamlit run dashboards/persistence_app.py
# Answers: does past performance predict future performance?
# Lets the user fiddle with parameters to test robustness of mean reversion finding.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
APP_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR    = PROJECT_DIR / "data" / "processed"
RAW_DIR     = PROJECT_DIR / "data" / "raw"

AUM_PANEL_FILE = str(DATA_DIR / "monthly_panel_with_aum.parquet")
ALPHA_FILE     = str(DATA_DIR / "trailing_3y_alphas.parquet")
FACTOR_FILE    = str(RAW_DIR  / "india_ff_momentum_monthly.csv")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.analysis import load_factors, build_decile_panel, run_regressions, summary_stats

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Persistence Dashboard',
    page_icon='📈',
    layout='wide',
)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    panel           = pd.read_parquet(AUM_PANEL_FILE)
    trailing_alphas = pd.read_parquet(ALPHA_FILE)
    factors         = load_factors(FACTOR_FILE)
    panel = panel[~panel['category'].str.contains('Hybrid', na=False)].copy()
    panel['cat_short'] = (panel['category']
        .str.replace('Equity Scheme - ', '', regex=False))
    return panel, trailing_alphas, factors

panel, trailing_alphas, factors = load_data()

# ── sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header('Parameters')

ranking_signal = st.sidebar.radio(
    'Ranking signal',
    options=['return', 'alpha'],
    format_func=lambda x: 'Trailing Return' if x == 'return' else 'Trailing 4-Factor Alpha',
    help='Alpha ranking requires 36 months of history so the sample is shorter',
)

if ranking_signal == 'return':
    ranking_months = st.sidebar.selectbox(
        'Ranking window (months)',
        options=[3, 6, 12, 24, 36],
        index=2,
        help='How many months of past return used to rank funds',
    )
else:
    ranking_months = 36
    st.sidebar.info('Alpha ranking always uses 36-month window')

holding_months = st.sidebar.selectbox(
    'Holding period (months)',
    options=[3, 6, 12, 24],
    index=2,
    help='How long to hold each decile portfolio after ranking',
)

categories     = ['All'] + sorted(panel['cat_short'].unique())
category       = st.sidebar.selectbox('Category filter', categories, index=0)

n_deciles      = st.sidebar.selectbox('Number of deciles', options=[5, 10], index=1)

# ── filter panel ──────────────────────────────────────────────────────────────
if category == 'All':
    filtered_panel = panel.copy()
else:
    filtered_panel = panel[panel['cat_short'] == category].copy()

# ── run analysis ──────────────────────────────────────────────────────────────
st.title('📈 Persistence Dashboard')
st.markdown(
    f"**Signal:** {'Trailing Return' if ranking_signal == 'return' else '4-Factor Alpha'} &nbsp;|&nbsp; "
    f"**Ranking:** {ranking_months}M &nbsp;|&nbsp; "
    f"**Holding:** {holding_months}M &nbsp;|&nbsp; "
    f"**Category:** {category} &nbsp;|&nbsp; "
    f"**Deciles:** {n_deciles}"
)

with st.spinner('Running decile analysis...'):
    try:
        decile_wide, meta = build_decile_panel(
            filtered_panel,
            factors,
            trailing_alphas=trailing_alphas if ranking_signal == 'alpha' else None,
            ranking_signal=ranking_signal,
            ranking_months=ranking_months,
            holding_months=holding_months,
            n_deciles=n_deciles,
        )
        results = run_regressions(decile_wide, factors)
    except ValueError as e:
        st.error(f"Could not run analysis: {e}")
        st.stop()

# ── summary metrics ───────────────────────────────────────────────────────────
spread = results.loc['Spread']

col1, col2, col3, col4 = st.columns(4)
col1.metric('Spread CAGR (D1-D10)',    f"{spread['CAGR']:+.2%}")
col2.metric('4F Alpha Spread',         f"{spread['ff4_alpha']:+.2%}")
col3.metric('t-statistic',             f"{spread['ff4_alpha_t']:+.2f}",
            help='|t| > 1.96 = significant at 5%')
col4.metric('Months / Rebalances',     f"{meta['n_months']} / {meta['n_rebalances']}")

# Interpretation
if spread['ff4_alpha_t'] < -1.96:
    st.error(f"📉 Significant mean reversion: past winners underperform past losers (t = {spread['ff4_alpha_t']:+.2f})")
elif spread['ff4_alpha_t'] > 1.96:
    st.success(f"📈 Significant persistence: past winners outperform past losers (t = {spread['ff4_alpha_t']:+.2f})")
else:
    st.info(f"No significant persistence or mean reversion (t = {spread['ff4_alpha_t']:+.2f})")

st.markdown('---')

# ── cumulative returns chart ───────────────────────────────────────────────────
st.subheader('Cumulative Returns by Decile')

fig, ax = plt.subplots(figsize=(14, 6))

cum = (1 + decile_wide.fillna(0)).cumprod()
cum.index = cum.index.to_timestamp()

cmap   = plt.cm.RdYlGn
n      = n_deciles
colors = {f'D{i+1}': cmap(1 - i/(n-1)) for i in range(n)}
colors['Spread'] = '#888888'

for col in decile_wide.columns:
    lw    = 2.2 if col == 'Spread' else 1.2
    ls    = '--' if col == 'Spread' else '-'
    alpha = 1.0 if col in ['D1', f'D{n}', 'Spread'] else 0.35
    label = col if col in ['D1', f'D{n}', 'Spread'] else '_nolegend_'
    ax.plot(cum.index, cum[col],
            color=colors.get(col, 'gray'),
            lw=lw, ls=ls, alpha=alpha, label=label)

ax.axhline(1, color='#cccccc', lw=0.8, ls=':')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}x'))
ax.set_ylabel('Growth of ₹1', fontsize=11)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, alpha=0.25)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── decile stats table ────────────────────────────────────────────────────────
st.subheader('Decile Statistics')

stats = decile_wide.apply(summary_stats).T
display_results = results[['CAGR', 'ff4_alpha', 'ff4_alpha_t',
                            'beta_SMB', 'beta_WML', 'R2']].copy()
display_results.columns = ['CAGR', '4F Alpha', 'Alpha t-stat',
                            'SMB Beta', 'WML Beta', 'R²']

for col in ['CAGR', '4F Alpha']:
    display_results[col] = display_results[col].map('{:+.2%}'.format)
for col in ['Alpha t-stat', 'SMB Beta', 'WML Beta']:
    display_results[col] = display_results[col].map('{:+.2f}'.format)
display_results['R²'] = display_results['R²'].map('{:.3f}'.format)

st.dataframe(display_results, use_container_width=True)

# ── alpha bar chart ───────────────────────────────────────────────────────────
st.subheader('4-Factor Alpha by Decile')
st.caption('* = |t| > 1.96')

decile_cols = [c for c in results.index if c.startswith('D')]
alphas      = results.loc[decile_cols, 'ff4_alpha'].values
tstats      = results.loc[decile_cols, 'ff4_alpha_t'].values
bar_colors  = ['#2166ac' if a >= 0 else '#d6604d' for a in alphas]

fig2, ax2 = plt.subplots(figsize=(12, 4))
bars = ax2.bar(decile_cols, alphas, color=bar_colors,
               edgecolor='white', linewidth=0.5)

for bar, t, a in zip(bars, tstats, alphas):
    if abs(t) >= 1.96:
        y_pos = a + (0.002 if a >= 0 else -0.006)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 '*', ha='center', fontsize=14, color='black')

ax2.axhline(0, color='black', lw=0.8)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
ax2.set_ylabel('Annualised 4-Factor Alpha')
ax2.grid(True, axis='y', alpha=0.25)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)
plt.close()