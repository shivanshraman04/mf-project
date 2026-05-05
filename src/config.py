#central parameter file for the mf-allocation project
#all magic numbers live here. Notebooks import them from this file

#usage in notebook if the notebook is in the /notebooks subfolder of the project
# import sys
# sys.path.append('..')
# from src.config import PANEL_START, EQUITY CATEGORIES

#if your notebook is in the project root, then:
# from src.config import PANEL_START, EQUITY CATEGORIES

# ── Date boundaries ──────────────────────────────────────────────────────────
#SEBI standardised mutual fund categories in October 2017. Data before that uses different category names that are not comparable to the
#  current system.
# Category system is reliable from here
CATEGORY_ANALYSIS_START = "2019-01"
#Use this for pure persistence analysis if you want to extend the analysis.  — usable for returnpersistence analysis only. Alpha persistence still requires factor data
# which is the binding constraint (see FACTOR_DATA_END).
PANEL_HISTORY_START = "2013-01"

#The IIMA factor data has a fixed end point that only updates when the authors release new data. Any analysis that uses factors must be
# clipped to this date or you will silently get NaN alphas for recent months
#Check for new updates at "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/"
#When new data arrives: replace the factor data file keeping the same name and path, update this constant, and re-run 04_alphas.ipynb
FACTOR_DATA_END = "2025-12"
#End data (last possible end-of-month available) for NAV data fetching. Update this when running a pipeline refresh
# Format must match MFAPI date parameter format: YYYY-MM-DD.
NAV_FETCH_END = "2026-03-31"



# ── MFAPI ─────────────────────────────────────────────────────────────────────
#
# MFAPI (https://api.mfapi.in) is a free, unofficial API that serves NAV
# history for all SEBI-registered mutual fund schemes. It has two endpoints
# we use:
#
#   GET /mf
#       Returns a list of all schemes with their scheme codes and names.
#       No parameters. Used in 01_universe.ipynb to build the initial list.
#
#   GET /mf/{scheme_code}
#       Returns full NAV history for a scheme, optionally filtered by date.
#       Parameters: startDate, endDate (format: DD-MM-YYYY, note: NOT YYYY-MM-DD)
#       Used in 02_panel.ipynb to fetch monthly returns.
#
#   GET /mf/{scheme_code}/latest
#       Returns only the most recent NAV and scheme metadata.
#       Used in 01_universe.ipynb to fetch category and fund house.
#
# MFAPI has no published rate limit. 0.05s delay (20 requests/second) has been
# stable across 618 funds. If you see connection errors or timeouts, increase
# API_DELAY_SECONDS to 0.1 or 0.2.

MFAPI_BASE = "https://api.mfapi.in/mf"
API_DELAY_SECONDS = 0.05
API_TIMEOUT_SECONDS = 15      # seconds before a request is abandoned
CHECKPOINT_EVERY = 50         # save progress to disk every N funds

# ── Universe definition ───────────────────────────────────────────────────────

# The 12 SEBI equity categories post-2019 reclassification.
# This list controls which funds are included in all analysis.
# Do not modify without also re-running 01_universe.ipynb, 02_panel.ipynb,
# and everything downstream — the entire panel is built from this filter.
# Note: 'Sectoral/ Thematic' has a slash and space — matches AMFI's exact string.
EQUITY_CATEGORIES = [
    'Equity Scheme - Large Cap Fund',
    'Equity Scheme - Mid Cap Fund',
    'Equity Scheme - Small Cap Fund',
    'Equity Scheme - Large & Mid Cap Fund',
    'Equity Scheme - Flexi Cap Fund',
    'Equity Scheme - Multi Cap Fund',
    'Equity Scheme - Focused Fund',
    'Equity Scheme - Value Fund',
    'Equity Scheme - Contra Fund',
    'Equity Scheme - Dividend Yield Fund',
    'Equity Scheme - ELSS',
    'Equity Scheme - Sectoral/ Thematic',
]

EQUITY_NAME_KEYWORDS = [
    'equity', 'opportunities', 'bluechip', 'blue chip',
    'midcap', 'mid cap', 'smallcap', 'small cap',
    'largecap', 'large cap', 'multicap', 'multi cap',
    'flexicap', 'flexi cap', 'elss', 'tax saver',
    'tax saving', 'diversified', 'contra', 'value',
    'focused', 'thematic', 'sectoral',
]

# Minimum months of return history a fund must have to be included in
# persistence analysis. Funds with fewer months are excluded from decile sorts.
# 10 months out of 12 = we tolerate up to 2 missing months in a ranking year.
MIN_MONTHS_IN_RANKING_YEAR = 10

# ── Factor model ──────────────────────────────────────────────────────────────

# Rolling window for computing trailing 4-factor alpha per fund-month. 36 months is the standard in the mutual fund persistence literature
# (Carhart 1997). Shorter windows produce noisier alpha estimates. Longer windows make the alpha less relevant to current fund behaviour.
# To run robustness checks at different windows, change this one constant and re-run 04_alphas.ipynb. Can change with this:
ALPHA_LOOKBACK_MONTHS = 36

# A fund must have at least this many valid observations within the rolling window to get an alpha estimate. Set to 80% of the window — funds 
# with more than 20% missing months get NaN rather than a noisy estimate.This updates automatically if you change ALPHA_LOOKBACK_MONTHS.
MIN_OBSERVATIONS = int(ALPHA_LOOKBACK_MONTHS * 0.8)

# Annual risk-free rate for summary statistics (Sharpe, category stats). This is NOT used in regressions — the factor CSV provides a monthly RF
# column drawn from the 91-day T-bill rate. This constant is only used for display calculations where we need a quick RF approximation.
RISK_FREE_ANNUAL = 0.06

# ── Persistence analysis ──────────────────────────────────────────────────────

# Number of deciles for portfolio sorts. 10 is standard. Reducing to 5 gives more funds per portfolio but less resolution at the tails where
# persistence effects are strongest.
N_DECILES = 10

# Decile labels: D1 = best prior performance, D10 = worst.  The spread portfolio is always long D1, short D10. (D1-D10)
SPREAD_LONG  = 'D1'
SPREAD_SHORT = f'D{N_DECILES}'

# Minimum funds needed in each decile at ranking date for that year to be valid.
# If any decile has fewer than this, the rebalance year is skipped entirely.
MIN_FUNDS_PER_DECILE = 5

## ── File paths ────────────────────────────────────────────────────────────────
#
# All paths are absolute, derived from the project root.
# This means notebooks can live anywhere in the project and paths will
# always resolve correctly — no need for os.chdir() or sys.path tricks.
#
# __file__ is the path to this config.py file (in src/).
# .parent gives src/, .parent again gives the project root.

from pathlib import Path

# Works both locally and on Streamlit Cloud
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Verify the path exists, if not try the Streamlit Cloud path
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = Path("/mount/src/mf-project")

PROJECT_ROOT = Path('/Users/shivanshraman/mf-project')

RAW_DATA_DIR       = str(PROJECT_ROOT / "data" / "raw")
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "data" / "processed")
AUM_DIR            = str(PROJECT_ROOT / "data" / "raw" / "aum")
OUTPUT_DIR         = str(PROJECT_ROOT / "outputs")

# Raw inputs
FACTOR_FILE        = str(PROJECT_ROOT / "data" / "raw" / "india_ff_momentum_monthly.csv")
ALL_SCHEMES_FILE   = str(PROJECT_ROOT / "data" / "raw" / "all_schemes.csv")
SCHEME_META_FILE   = str(PROJECT_ROOT / "data" / "raw" / "scheme_meta.csv")

# Processed data
EQUITY_UNIVERSE_FILE = str(PROJECT_ROOT / "data" / "processed" / "equity_universe.parquet")
MONTHLY_PANEL_FILE   = str(PROJECT_ROOT / "data" / "processed" / "monthly_panel_enriched.parquet")
AUM_PANEL_FILE       = str(PROJECT_ROOT / "data" / "processed" / "monthly_panel_with_aum.parquet")
ALPHA_FILE           = str(PROJECT_ROOT / "data" / "processed" / "trailing_3y_alphas.parquet")
EXTENDED_PANEL_FILE  = str(PROJECT_ROOT / "data" / "processed" / "extended_panel.parquet")

# Output artifacts
PERSISTENCE_RETURN_FILE = str(PROJECT_ROOT / "outputs" / "persistence_return_sorted.csv")
PERSISTENCE_ALPHA_FILE  = str(PROJECT_ROOT / "outputs" / "persistence_alpha_sorted.csv")
CATEGORY_STATS_FILE     = str(PROJECT_ROOT / "outputs" / "category_stats_aum_weighted.csv")

#Extended Universe
EXTENDED_UNIVERSE_FILE = str(PROJECT_ROOT / "data" / "processed" / "extended_equity_universe.parquet")
EXTENDED_PANEL_FILE    = str(PROJECT_ROOT / "data" / "processed" / "extended_panel.parquet")