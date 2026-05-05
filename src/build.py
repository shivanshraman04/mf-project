# src/build.py
#
# Functions for constructing the monthly returns panel and integrating AUM data.
# Called by notebooks — no fetching logic here, just panel construction.

import pandas as pd
import numpy as np
import re
import glob
from src.config import PROCESSED_DATA_DIR, AUM_DIR


def build_panel(nav_frames, universe):
    """
    Concatenate per-fund NAV DataFrames and merge metadata.

    Args:
        nav_frames: list of DataFrames returned by get_nav_history()
        universe:   DataFrame with columns scheme_code, scheme_name,
                    category, fund_house — from equity_universe.parquet

    Returns:
        DataFrame with columns:
            scheme_code, year_month, date, nav, monthly_return,
            scheme_name, category, fund_house
    """
    if isinstance(nav_frames, pd.DataFrame):
        panel = nav_frames.copy()
    else:
        panel = pd.concat(nav_frames, ignore_index=True)

    # These run regardless of which branch above was taken
    panel = panel.drop_duplicates(subset=['scheme_code', 'year_month'])

    panel = panel.merge(
        universe[['scheme_code', 'scheme_name', 'category', 'fund_house']],
        on='scheme_code',
        how='left'
    )

    return panel.sort_values(['scheme_code', 'year_month']).reset_index(drop=True)

def parse_amfi_aum_file(filepath):
    """
    Parse one AMFI quarterly AUM Excel file.

    AMFI publishes AUM data quarterly. Each file covers one quarter and
    contains scheme codes, names, and average AUM in lakhs.
    The file format has the quarter and year in the first row as a title,
    followed by a header row, then data rows.

    Returns a DataFrame with columns:
        scheme_code, aaum_lakhs, quarter_label, year, month_start, month_end
    """
    df = pd.read_excel(filepath, header=None)
    title = str(df.iloc[0, 0])

    quarter_month_map = {
        'January - March':    (1, 3),
        'April - June':       (4, 6),
        'July - September':   (7, 9),
        'October - December': (10, 12),
    }

    year = month_start = month_end = quarter_label = None
    for q_text, (m_start, m_end) in quarter_month_map.items():
        if q_text in title:
            year_match = re.search(r'(\d{4})', title)
            if year_match:
                year = int(year_match.group(1))
                quarter_label = f"{q_text} {year}"
                month_start, month_end = m_start, m_end
            break

    if year is None:
        raise ValueError(f"Could not parse quarter/year from: {title}")

    data_rows = df.iloc[2:].copy()
    data_rows.columns = ['amfi_code', 'scheme_name', 'aaum_lakhs', 'fof_domestic']
    data_rows = data_rows.reset_index(drop=True)

    def is_numeric_code(val):
        try:
            int(float(str(val)))
            return True
        except Exception:
            return False

    clean = data_rows[data_rows['amfi_code'].apply(is_numeric_code)].copy()
    clean['scheme_code'] = clean['amfi_code'].apply(lambda x: int(float(str(x))))
    clean['aaum_lakhs']  = pd.to_numeric(clean['aaum_lakhs'], errors='coerce')
    clean['quarter_label'] = quarter_label
    clean['year']          = year
    clean['month_start']   = month_start
    clean['month_end']     = month_end

    return (clean[['scheme_code', 'aaum_lakhs', 'quarter_label',
                   'year', 'month_start', 'month_end']]
            .dropna(subset=['aaum_lakhs'])
            .reset_index(drop=True))


def build_aum_panel(aum_dir=AUM_DIR):
    """
    Parse all quarterly AUM Excel files and expand to monthly rows.

    AMFI reports AUM quarterly. We expand each quarter's figure to three
    monthly rows — one per month in the quarter — using the same AUM value
    for all three months. This is the standard approach since intra-quarter
    AUM is not published.

    Returns a DataFrame with columns:
        scheme_code, year_month, aaum_lakhs
    """
    aum_files = sorted(glob.glob(f"{aum_dir}/*.xlsx"))
    if not aum_files:
        raise FileNotFoundError(f"No AUM Excel files found in {aum_dir}")

    quarterly_frames = []
    for fp in aum_files:
        try:
            q = parse_amfi_aum_file(fp)
            quarterly_frames.append(q)
            print(f"  Parsed: {q['quarter_label'].iloc[0]} ({len(q)} schemes)")
        except Exception as e:
            print(f"  FAILED: {fp} → {e}")

    combined = pd.concat(quarterly_frames, ignore_index=True)

    # Expand each quarter to monthly rows
    monthly_rows = []
    for _, row in combined.iterrows():
        for month in range(int(row['month_start']), int(row['month_end']) + 1):
            monthly_rows.append({
                'scheme_code': row['scheme_code'],
                'year_month':  pd.Period(f"{int(row['year'])}-{month:02d}", freq='M'),
                'aaum_lakhs':  row['aaum_lakhs'],
            })

    aum_monthly = pd.DataFrame(monthly_rows)
    return aum_monthly.sort_values(['scheme_code', 'year_month']).reset_index(drop=True)


def merge_aum(panel, aum_monthly):
    """
    Merge lagged AUM onto the returns panel.

    AUM is lagged by one month before merging to prevent look-ahead bias —
    the AUM figure for month t is attached to month t+1's return. This
    ensures we only use AUM that was known before the period being measured.

    Args:
        panel:       enriched returns panel from build_panel()
        aum_monthly: monthly AUM panel from build_aum_panel()

    Returns:
        panel with additional column: aaum_lakhs_lagged
    """
    aum_lagged = aum_monthly.copy()
    aum_lagged['year_month']  = aum_lagged['year_month'] + 1
    aum_lagged = aum_lagged.rename(columns={'aaum_lakhs': 'aaum_lakhs_lagged'})

    return panel.merge(
        aum_lagged[['scheme_code', 'year_month', 'aaum_lakhs_lagged']],
        on=['scheme_code', 'year_month'],
        how='left'
    )