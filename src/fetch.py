# src/fetch.py
#
# Functions for fetching data from MFAPI.
# Called by notebooks — no analysis logic here, just data retrieval.

import requests
import pandas as pd
from src.config import MFAPI_BASE, API_DELAY_SECONDS, API_TIMEOUT_SECONDS


def get_all_schemes():
    """
    Fetch the full list of schemes from MFAPI.
    Returns a DataFrame with columns: schemeCode, schemeName.
    This is the raw universe before any filtering.
    """
    r = requests.get(MFAPI_BASE, timeout=30)
    return pd.DataFrame(r.json())


def is_direct_growth(name):
    """
    Returns True if a scheme name represents a direct-growth plan.
    
    The tricky case: 'dividend' appears in both "Dividend Yield" (a fund 
    category, acceptable) and "Dividend Payout" (a plan type, excluded).
    Solution: strip 'dividend yield' from the name first, then check for
    any remaining 'dividend' which would indicate a payout plan.
    """
    name_lower = name.lower()

    if 'direct' not in name_lower or 'growth' not in name_lower:
        return False
    if 'idcw' in name_lower:
        return False
    if 'payout' in name_lower or 'reinvest' in name_lower:
        return False

    # Remove 'dividend yield' (category name) before checking for dividend
    if 'dividend' in name_lower.replace('dividend yield', ''):
        return False

    return True


def get_scheme_meta(scheme_code):
    """
    Fetch metadata for a single scheme: name, category, type, fund house.
    Uses the /latest endpoint which is a small payload (no full NAV history).
    Returns a dict, or None if the request fails or status is not SUCCESS.
    """
    try:
        r = requests.get(
            f"{MFAPI_BASE}/{scheme_code}/latest",
            timeout=API_TIMEOUT_SECONDS
        )
        payload = r.json()
        if payload.get('status') != 'SUCCESS':
            return None
        meta = payload.get('meta', {})
        return {
            'scheme_code': scheme_code,
            'scheme_name': meta.get('scheme_name'),
            'category':    meta.get('scheme_category'),
            'scheme_type': meta.get('scheme_type'),
            'fund_house':  meta.get('fund_house'),
        }
    except Exception:
        return None

def get_nav_history(scheme_code, start_date, end_date):
    """
    Fetch NAV history for a single scheme and return monthly returns.

    Steps:
        1. Fetch daily NAV from MFAPI between start_date and end_date
        2. Take the last NAV of each calendar month
        3. Compute month-over-month percentage change

    Args:
        scheme_code: integer scheme code
        start_date:  string in YYYY-MM-DD format (converted internally to DD-MM-YYYY for MFAPI)
        end_date:    string in YYYY-MM-DD format

    Returns a DataFrame with columns:
        scheme_code, year_month, date, nav, monthly_return
    Returns None if fetch fails or fewer than 2 months of data.

    Note on date format: MFAPI expects DD-MM-YYYY but we accept YYYY-MM-DD
    here because that is the standard Python/pandas format. Conversion happens
    inside this function so callers never need to think about it.
    """
    start_api = pd.to_datetime(start_date).strftime('%d-%m-%Y')
    end_api   = pd.to_datetime(end_date).strftime('%d-%m-%Y')

    try:
        r = requests.get(
            f"{MFAPI_BASE}/{scheme_code}",
            params={'startDate': start_api, 'endDate': end_api},
            timeout=API_TIMEOUT_SECONDS
        )
        payload = r.json()
    except Exception:
        return None

    if payload.get('status') != 'SUCCESS':
        return None
    if not payload.get('data'):
        return None

    df = pd.DataFrame(payload['data'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav']  = pd.to_numeric(df['nav'], errors='coerce')
    df = df.dropna(subset=['nav']).sort_values('date').reset_index(drop=True)

    if len(df) < 2:
        return None

    df['year_month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('year_month').last().reset_index()
    monthly['monthly_return'] = monthly['nav'].pct_change()
    monthly['scheme_code']    = scheme_code

    # MFAPI sometimes ignores date parameters — filter explicitly
    start_period = pd.to_datetime(start_date).to_period('M')
    end_period   = pd.to_datetime(end_date).to_period('M')
    monthly = monthly[
        (monthly['year_month'] >= start_period) &
        (monthly['year_month'] <= end_period)
    ]

    if len(monthly) < 2:
        return None

    return monthly[['scheme_code', 'year_month', 'date', 'nav', 'monthly_return']]