"""
data_refresh.py - Fetch and validate macroeconomic data from FRED API.

This module handles:
- Downloading CPI, WPI, Interest Rate, USD/INR, and Brent Crude data
- Validating data quality (null checks, anomaly detection)
- Merging data into inflation_dataset.csv
- Logging refresh operations to data/refresh_log.json
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
if not FRED_API_KEY:
    raise RuntimeError('FRED_API_KEY environment variable is required')
FRED_BASE_URL = 'https://api.stlouisfed.org/fred/series/obs'

SERIES = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'rate': 'IRSTCB01INM156N',
    'fx': 'DEXINUS',
    'brent': 'DCOILBRENTEU',
}

DATA_DIR = Path('data')
MERGED_CSV = DATA_DIR / 'inflation_dataset.csv'
REFRESH_LOG = DATA_DIR / 'refresh_log.json'


def fetch_fred_series(series_id, api_key=None):
    api_key = api_key or FRED_API_KEY
    params = {'series_id': series_id, 'api_key': api_key, 'filetype': 'json'}
    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'observations' not in data:
            raise ValueError(f'No observations for series {series_id}')
        rows = []
        for obs in data['observations']:
            if obs['value'] == 'NULL':
                continue
            rows.append({'date': obs['date'], 'value': float(obs['value'])})
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f'No valid data for series {series_id}')
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f'FRED API error for {series_id}: {e}')


def validate_cpi(df):
    issues = []
    if df['value'].isna().any():
        issues.append('Contains null values')
    if df['value'].min() < 80 or df['value'].max() > 250:
        issues.append(f'Value out of range: min={df["value"].min():.2f}, max={df["value"].max():.2f}')
    df = df.sort_values('date')
    df['mom_change'] = df['value'].pct_change() * 100
    extreme = df[abs(df['mom_change']) > 5]
    if not extreme.empty:
        issues.append(f'{len(extreme)} extreme MoM changes (>5%)')
    return issues


def validate_wpi(df):
    issues = []
    if df['value'].isna().any():
        issues.append('Contains null values')
    if (df['value'] <= 0).any():
        issues.append('Contains non-positive values')
    return issues


def validate_rate(df):
    issues = []
    if df['value'].isna().any():
        issues.append('Contains null values')
    if df['value'].min() < 0 or df['value'].max() > 20:
        issues.append(f'Value out of range: min={df["value"].min():.2f}, max={df["value"].max():.2f}')
    return issues


def validate_fx(df):
    issues = []
    if df['value'].isna().any():
        issues.append('Contains null values')
    df = df.sort_values('date')
    df['day_change'] = df['value'].pct_change() * 100
    extreme = df[abs(df['day_change']) > 2]
    if not extreme.empty:
        issues.append(f'{len(extreme)} extreme daily changes (>2%)')
    return issues


def validate_brent(df):
    issues = []
    if df['value'].isna().any():
        issues.append('Contains null values')
    df = df.sort_values('date')
    df['day_change'] = df['value'].pct_change() * 100
    extreme = df[abs(df['day_change']) > 10]
    if not extreme.empty:
        issues.append(f'{len(extreme)} extreme daily changes (>10%)')
    return issues


def refresh_cpi(api_key=None):
    print('Fetching CPI data...')
    df = fetch_fred_series(SERIES['cpi'], api_key=api_key)
    issues = validate_cpi(df)
    if issues:
        print(f'  Warnings: {issues}')
    return df, issues


def refresh_wpi(api_key=None):
    print('Fetching WPI data...')
    df = fetch_fred_series(SERIES['wpi'], api_key=api_key)
    issues = validate_wpi(df)
    if issues:
        print(f'  Warnings: {issues}')
    return df, issues


def refresh_rate(api_key=None):
    print('Fetching Interest Rate data...')
    df = fetch_fred_series(SERIES['rate'], api_key=api_key)
    issues = validate_rate(df)
    if issues:
        print(f'  Warnings: {issues}')
    return df, issues


def refresh_fx(api_key=None):
    print('Fetching USD/INR data...')
    df = fetch_fred_series(SERIES['fx'], api_key=api_key)
    issues = validate_fx(df)
    if issues:
        print(f'  Warnings: {issues}')
    return df, issues


def refresh_brent(api_key=None):
    print('Fetching Brent Crude data...')
    df = fetch_fred_series(SERIES['brent'], api_key=api_key)
    issues = validate_brent(df)
    if issues:
        print(f'  Warnings: {issues}')
    return df, issues


def merge_and_save(cpi_df, wpi_df, rate_df, fx_df, brent_df):
    cpi_df = cpi_df.rename(columns={'date': 'observation_date', 'value': 'INDCPIALLMINMEI'})
    wpi_df = wpi_df.rename(columns={'date': 'observation_date', 'value': 'WPIATT01INM661N'})
    rate_df = rate_df.rename(columns={'date': 'observation_date', 'value': 'INTDSRINM193N'})
    fx_df = fx_df.rename(columns={'date': 'observation_date', 'value': 'DEXINUS'})
    brent_df = brent_df.rename(columns={'date': 'observation_date', 'value': 'Average of DCOILBRENTEU'})
    merged = cpi_df[['observation_date', 'INDCPIALLMINMEI']]
    merged = merged.merge(wpi_df[['observation_date', 'WPIATT01INM661N']], on='observation_date', how='outer')
    merged = merged.merge(rate_df[['observation_date', 'INTDSRINM193N']], on='observation_date', how='outer')
    merged = merged.merge(fx_df[['observation_date', 'DEXINUS']], on='observation_date', how='outer')
    merged = merged.merge(brent_df[['observation_date', 'Average of DCOILBRENTEU']], on='observation_date', how='outer')
    merged['observation_date'] = pd.to_datetime(merged['observation_date'])
    merged = merged.sort_values('observation_date').reset_index(drop=True)
    merged = merged.ffill().bfill()
    merged = merged.set_index('observation_date').resample('ME').last().reset_index()
    merged = merged.ffill()
    merged.to_csv(MERGED_CSV, index=False)
    print(f'Saved merged dataset to {MERGED_CSV}')
    return merged


def log_refresh(results):
    log_entry = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'sources': results}
    if REFRESH_LOG.exists():
        with open(REFRESH_LOG, 'r') as f:
            log_data = json.load(f)
        if not isinstance(log_data, list):
            log_data = [log_data]
        log_data.append(log_entry)
    else:
        log_data = [log_entry]
    with open(REFRESH_LOG, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f'Logged refresh to {REFRESH_LOG}')


def run_refresh(api_key=None):
    print('=' * 60)
    print('VATICMACRO DATA REFRESH')
    print('=' * 60)
    results = {}
    try:
        cpi_df, cpi_issues = refresh_cpi(api_key=api_key)
        results['cpi'] = {'status': 'success', 'rows': len(cpi_df), 'last_date': cpi_df['date'].max().strftime('%Y-%m-%d'), 'issues': cpi_issues}
    except Exception as e:
        results['cpi'] = {'status': 'error', 'message': str(e)}
        cpi_df = None
    try:
        wpi_df, wpi_issues = refresh_wpi(api_key=api_key)
        results['wpi'] = {'status': 'success', 'rows': len(wpi_df), 'last_date': wpi_df['date'].max().strftime('%Y-%m-%d'), 'issues': wpi_issues}
    except Exception as e:
        results['wpi'] = {'status': 'error', 'message': str(e)}
        wpi_df = None
    try:
        rate_df, rate_issues = refresh_rate(api_key=api_key)
        results['rate'] = {'status': 'success', 'rows': len(rate_df), 'last_date': rate_df['date'].max().strftime('%Y-%m-%d'), 'issues': rate_issues}
    except Exception as e:
        results['rate'] = {'status': 'error', 'message': str(e)}
        rate_df = None
    try:
        fx_df, fx_issues = refresh_fx(api_key=api_key)
        results['fx'] = {'status': 'success', 'rows': len(fx_df), 'last_date': fx_df['date'].max().strftime('%Y-%m-%d'), 'issues': fx_issues}
    except Exception as e:
        results['fx'] = {'status': 'error', 'message': str(e)}
        fx_df = None
    try:
        brent_df, brent_issues = refresh_brent(api_key=api_key)
        results['brent'] = {'status': 'success', 'rows': len(brent_df), 'last_date': brent_df['date'].max().strftime('%Y-%m-%d'), 'issues': brent_issues}
    except Exception as e:
        results['brent'] = {'status': 'error', 'message': str(e)}
        brent_df = None
    if cpi_df is not None and wpi_df is not None:
        try:
            merged = merge_and_save(cpi_df, wpi_df, rate_df, fx_df, brent_df)
            results['merged'] = {'status': 'success', 'rows': len(merged), 'date_range': f'{merged["observation_date"].min().strftime("%Y-%m-%d")} to {merged["observation_date"].max().strftime("%Y-%m-%d")}'}
        except Exception as e:
            results['merged'] = {'status': 'error', 'message': str(e)}
    else:
        results['merged'] = {'status': 'skipped', 'message': 'Core sources missing'}
    log_refresh(results)
    print('=' * 60)
    print('REFRESH COMPLETE')
    print('=' * 60)
    return results


if __name__ == '__main__':
    import sys
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    run_refresh(api_key=api_key)
