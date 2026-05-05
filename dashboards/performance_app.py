# dashboards/performance_app.py
#
# Run with: streamlit run dashboards/performance_app.py
# Answers: which funds have been consistent and good within their category?

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter

from src.config import AUM_PANEL_FILE, FACTOR_FILE
from src.analysis import load_factors

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Fund Performance Dashboard',
    page_icon='📊',
    layout='wide',
)

# ── load data (cached so it only runs once) ───────────────────────────────────
@st.cache_data
def load_data():
    panel   = pd.read_parquet(AUM_PANEL_FILE)
    factors = load_factors(FACTOR_FILE)

    panel['cat_short'] = (panel['category']
        .str.replace('Equity Scheme - ', '', regex=False))

    panel['fund_short'] = (panel['scheme_name']
        .str.replace(' - Direct Plan - Growth', '', regex=False)
        .str.replace(' - Direct - Growth', '', regex=False)
        .str.replace(' Direct Plan Growth', '', regex=False))

    return panel, factors

panel, factors = load_data()

# ── sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header('Parameters')

categories = ['All'] + sorted(panel['cat_short'].unique())
category   = st.sidebar.selectbox('Category', categories, index=0)

all_months = sorted(panel['year_month'].unique())
min_date   = str(all_months[0])
max_date   = str(all_months[-1])

start_period = st.sidebar.select_slider(
    'Start period',
    options=[str(m) for m in all_months],
    value='2019-01',
)
end_period = st.sidebar.select_slider(
    'End period',
    options=[str(m) for m in all_months],
    value=max_date,
)

interval_months = st.sidebar.selectbox(
    'Interval length (months)',
    options=[3, 6, 12, 24],
    index=2,
)

min_coverage = st.sidebar.slider(
    'Min interval coverage',
    min_value=0.25,
    max_value=1.0,
    value=0.50,
    step=0.05,
    format='%.0f%%',
    help='Fund must appear in this fraction of intervals to be included',
)

st.sidebar.markdown('---')
fund_search = st.sidebar.text_input(
    'Fund-specific view (type fund name)',
    value='',
    placeholder='e.g. SBI Small Cap',
)

# ── filter panel ──────────────────────────────────────────────────────────────
start = pd.Period(start_period, freq='M')
end   = pd.Period(end_period,   freq='M')

if category == 'All':
    cat_panel = panel[
        (panel['year_month'] >= start) &
        (panel['year_month'] <= end)
    ].copy()
else:
    cat_panel = panel[
        (panel['cat_short'] == category) &
        (panel['year_month'] >= start) &
        (panel['year_month'] <= end)
    ].copy()

# ── compute percentile ranks per interval ─────────────────────────────────────
all_months_cat = sorted(cat_panel['year_month'].unique())
n_intervals    = len(all_months_cat) // interval_months

interval_results = []

for i in range(n_intervals):
    iv_start = all_months_cat[i * interval_months]
    iv_end   = all_months_cat[min((i + 1) * interval_months - 1,
                                   len(all_months_cat) - 1)]

    iv_data = cat_panel[
        (cat_panel['year_month'] >= iv_start) &
        (cat_panel['year_month'] <= iv_end)
    ]

    fund_returns = (
        iv_data
        .groupby('scheme_code')['monthly_return']
        .apply(lambda r: (1 + r.dropna()).prod() - 1
               if r.notna().sum() >= interval_months * 0.7
               else np.nan)
        .dropna()
    )

    if len(fund_returns) < 5:
        continue

    pct_rank = fund_returns.rank(pct=True) * 100

    for code, rank in pct_rank.items():
        interval_results.append({
            'scheme_code':    code,
            'interval':       i + 1,
            'interval_label': f"{iv_start}→{iv_end}",
            'cum_return':     fund_returns[code],
            'pct_rank':       rank,
        })

interval_df = pd.DataFrame(interval_results)

# ── compute fund stats ────────────────────────────────────────────────────────
if len(interval_df) == 0:
    st.error("No data for selected parameters. Try a wider date range or shorter interval.")
    st.stop()

n_intervals_total = interval_df['interval'].nunique()
min_intervals     = max(1, int(n_intervals_total * min_coverage))

fund_stats = (
    interval_df
    .groupby('scheme_code')
    .agg(
        avg_rank    = ('pct_rank', 'mean'),
        sd_rank     = ('pct_rank', 'std'),
        n_intervals = ('pct_rank', 'count'),
        avg_return  = ('cum_return', 'mean'),
    )
    .reset_index()
)

fund_stats = fund_stats[fund_stats['n_intervals'] >= min_intervals].copy()

names      = panel[['scheme_code', 'fund_short']].drop_duplicates()
fund_stats = fund_stats.merge(names, on='scheme_code', how='left')
fund_stats = fund_stats.sort_values('avg_rank', ascending=False).reset_index(drop=True)
fund_stats.index += 1

# ── main page ─────────────────────────────────────────────────────────────────
st.title('📊 Fund Performance Dashboard')
st.markdown(
    f"**Category:** {category} &nbsp;|&nbsp; "
    f"**Period:** {start_period} → {end_period} &nbsp;|&nbsp; "
    f"**Interval:** {interval_months} months &nbsp;|&nbsp; "
    f"**Funds shown:** {len(fund_stats)}"
)

# ── rank table ────────────────────────────────────────────────────────────────
st.subheader('Rank Table')
st.caption('Avg Rank: 100 = always best in category. SD Rank: lower = more consistent.')

display = fund_stats[['fund_short', 'avg_rank', 'sd_rank',
                       'n_intervals', 'avg_return']].copy()
display.columns = ['Fund', 'Avg Rank', 'SD Rank', 'Intervals', 'Avg Return/Interval']
display['Avg Rank']           = display['Avg Rank'].round(1)
display['SD Rank']            = display['SD Rank'].round(1)
display['Avg Return/Interval'] = display['Avg Return/Interval'].map('{:.2%}'.format)

st.dataframe(display, use_container_width=True, height=400)

# ── scatter plot ──────────────────────────────────────────────────────────────
st.subheader('Performance vs Consistency')
st.caption('Top right = high average rank. Low SD = consistent. Best funds: top right, low SD.')

fig, ax = plt.subplots(figsize=(12, 6))

sc = ax.scatter(
    fund_stats['avg_rank'],
    fund_stats['sd_rank'],
    alpha=0.7, s=60,
    c=fund_stats['avg_rank'],
    cmap='RdYlGn',
    vmin=0, vmax=100,
    edgecolors='white', linewidths=0.5,
)

top_avg  = fund_stats.nlargest(5, 'avg_rank')
top_cons = fund_stats.nsmallest(5, 'sd_rank')
to_label = pd.concat([top_avg, top_cons]).drop_duplicates('scheme_code')

for _, row in to_label.iterrows():
    ax.annotate(
        row['fund_short'][:35],
        (row['avg_rank'], row['sd_rank']),
        fontsize=7, xytext=(5, 5),
        textcoords='offset points', alpha=0.9,
    )

ax.axvline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(fund_stats['sd_rank'].median(),
           color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Average Percentile Rank (100 = always best)', fontsize=11)
ax.set_ylabel('SD of Percentile Rank (lower = more consistent)', fontsize=11)
ax.grid(True, alpha=0.2)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.colorbar(sc, ax=ax, label='Average Percentile Rank')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── fund-specific view ────────────────────────────────────────────────────────
if fund_search:
    st.markdown('---')
    st.subheader(f'Fund View: {fund_search}')

    match = panel[panel['fund_short'].str.contains(fund_search, case=False, na=False)]

    if len(match) == 0:
        st.warning(f"No fund found matching '{fund_search}'")
    else:
        fund_code = match['scheme_code'].iloc[0]
        fund_name = match['fund_short'].iloc[0]
        fund_data = panel[
            (panel['scheme_code'] == fund_code) &
            (panel['year_month'] >= start) &
            (panel['year_month'] <= end)
        ].sort_values('year_month')

        fund_intervals = interval_df[interval_df['scheme_code'] == fund_code]

        col1, col2, col3, col4 = st.columns(4)

        if len(fund_intervals) > 0:
            col1.metric('Avg Percentile Rank', f"{fund_intervals['pct_rank'].mean():.1f}")
            col2.metric('SD of Rank',          f"{fund_intervals['pct_rank'].std():.1f}")
            col3.metric('Intervals Covered',   f"{len(fund_intervals)}/{n_intervals_total}")

        # CAPM alpha
        common = fund_data['year_month'].isin(factors.index)
        fund_f = fund_data[common].copy()
        fac_f  = factors.loc[fund_f['year_month']]

        if len(fund_f) >= 24:
            y  = fund_f['monthly_return'].values
            rf = fac_f['RF'].values
            mf = fac_f['MF'].values
            ye = y - rf
            X  = sm.add_constant(pd.Series(mf, name='MF'))
            m  = sm.OLS(ye, X).fit()
            capm_alpha = m.params['const'] * 12
            capm_t     = m.tvalues['const']
            col4.metric('CAPM Alpha (ann.)',
                        f"{capm_alpha:+.2%}",
                        f"t = {capm_t:+.2f}")

        # Charts
        fig2, axes = plt.subplots(1, 2, figsize=(14, 4))
        # Cumulative return
        ax1  = axes[0]
        r    = fund_data['monthly_return'].fillna(0)
        cum  = (1 + r).cumprod()
        dates = fund_data['year_month'].dt.to_timestamp()
        ax1.plot(dates, cum, linewidth=2, color='#1f4e79')
        ax1.fill_between(dates, 1, cum, alpha=0.1, color='#1f4e79')
        ax1.axhline(1, color='gray', linestyle='--', linewidth=0.8)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}x'))
        ax1.set_title(f'Cumulative Return\n{fund_name[:50]}',
                      fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.25)
        for sp in ['top', 'right']:
            ax1.spines[sp].set_visible(False)

        # Percentile rank by interval
        ax2 = axes[1]
        if len(fund_intervals) > 0:
            colors = ['#2ecc71' if r >= 50 else '#e74c3c'
                      for r in fund_intervals['pct_rank']]
            ax2.bar(fund_intervals['interval'],
                    fund_intervals['pct_rank'],
                    color=colors, edgecolor='white', linewidth=0.5)
            ax2.axhline(50, color='gray', linestyle='--', linewidth=0.8)
            ax2.set_ylim(0, 100)
            ax2.set_xlabel(f'Interval ({interval_months}-month periods)')
            ax2.set_ylabel('Percentile Rank (100 = best)')
            ax2.set_title(f'Percentile Rank by Interval\n{fund_name[:50]}',
                          fontsize=10, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.25)
            for sp in ['top', 'right']:
                ax2.spines[sp].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()