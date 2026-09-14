"""
data_refresh.py - Fetch and validate macroeconomic data from FRED API.

This module handles:
- Downloading CPI, WPI, Interest Rate, USD/INR, and Brent Crude data
- Validating data quality (null checks, anomaly detection)
- Merging data into inflation_dataset.csv
- Logging refresh operations to data/refresh_log.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "cpi": "INDCPIALLMINMEI",
    "wpi": "WPIATT01INM661N",
    "rate": "INTDSRINM193N",
    "fx": "DEXINUS",
    "brent": "DCOILBRENTEU",
}

DATA_DIR = Path("data")
MERGED_CSV = DATA_DIR / "inflation_dataset.csv"
REFRESH_LOG = DATA_DIR / "refresh_log.json"


def fetch_fred_series(series_id, api_key=None):
    api_key = api_key or FRED_API_KEY
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "observations" not in data:
            raise ValueError(f"No observations for series {series_id}")
        rows = []
        for obs in data["observations"]:
            val = obs["value"]
            if val == "NULL" or val == ".":
                continue
            try:
                rows.append({"date": obs["date"], "value": float(val)})
            except ValueError:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"No valid data for series {series_id}")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"FRED API error for {series_id}: {e}") from e


def fetch_rbi_dbie_series(indicator: str = "WPI") -> pd.DataFrame | None:
    """Best-effort RBI DBIE fallback when FRED is unavailable.

    Returns DataFrame with date/value or None if RBI fetch fails.
    Never raises — caller decides fallback behavior.
    """
    try:
        # Placeholder: RBI DBIE requires manual scraping; for now return None
        # Future: scrape https://data.rbi.org.in/DBIE/dbie.rbi?site=statistics
        # This stub preserves the COLUMN_MAP contract without breaking CI.
        return None
    except Exception:
        return None


def validate_cpi(df):
    issues = []
    if df["value"].isna().any():
        issues.append("Contains null values")
    if df["value"].min() < 80 or df["value"].max() > 250:
        issues.append(
            f"Value out of range: min={df['value'].min():.2f}, max={df['value'].max():.2f}"
        )
    df = df.sort_values("date")
    df["mom_change"] = df["value"].pct_change() * 100
    extreme = df[abs(df["mom_change"]) > 5]
    if not extreme.empty:
        issues.append(f"{len(extreme)} extreme MoM changes (>5%)")
    return issues


def validate_wpi(df):
    issues = []
    if df["value"].isna().any():
        issues.append("Contains null values")
    if (df["value"] <= 0).any():
        issues.append("Contains non-positive values")
    return issues


def validate_rate(df):
    issues = []
    if df["value"].isna().any():
        issues.append("Contains null values")
    if df["value"].min() < 0 or df["value"].max() > 20:
        issues.append(
            f"Value out of range: min={df['value'].min():.2f}, max={df['value'].max():.2f}"
        )
    return issues


def validate_fx(df):
    issues = []
    if df["value"].isna().any():
        issues.append("Contains null values")
    df = df.sort_values("date")
    df["day_change"] = df["value"].pct_change() * 100
    extreme = df[abs(df["day_change"]) > 2]
    if not extreme.empty:
        issues.append(f"{len(extreme)} extreme daily changes (>2%)")
    return issues


def validate_brent(df):
    issues = []
    if df["value"].isna().any():
        issues.append("Contains null values")
    df = df.sort_values("date")
    df["day_change"] = df["value"].pct_change() * 100
    extreme = df[abs(df["day_change"]) > 10]
    if not extreme.empty:
        issues.append(f"{len(extreme)} extreme daily changes (>10%)")
    return issues


def refresh_cpi(api_key=None):
    print("Fetching CPI data from Trading Economics aggregator...")
    import urllib.request
    from pathlib import Path

    from bs4 import BeautifulSoup

    cpi_path = Path("data/INDCPIALLMINMEI (Consumer Price Index).csv")
    df = pd.read_csv(cpi_path)
    date_col = "observation_date" if "observation_date" in df.columns else "Date"
    df["date"] = pd.to_datetime(df[date_col])
    val_col = "INDCPIALLMINMEI" if "INDCPIALLMINMEI" in df.columns else "value"
    df = df[["date", val_col]].rename(columns={val_col: "value"})

    last_date = df["date"].max()

    try:
        url = "https://tradingeconomics.com/india/consumer-price-index-cpi"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")

        for t in tables:
            if "table-hover" in t.get("class", []):
                for r in t.find_all("tr"):
                    tds = [td.text.strip() for td in r.find_all(["th", "td"])]
                    if len(tds) >= 5 and tds[0] == "Inflation Rate MoM":
                        mom_pct = float(tds[1])
                        ref_date_str = tds[4]  # e.g. "May 2026"

                        # Convert "May 2026" to month-end date for alignment
                        ref_date = pd.to_datetime(ref_date_str) + pd.offsets.MonthEnd(0)

                        last_date_dt = pd.to_datetime(last_date)

                        if (ref_date.year > last_date_dt.year) or (
                            ref_date.year == last_date_dt.year
                            and ref_date.month > last_date_dt.month
                        ):
                            print(
                                f"  Found new CPI data for {ref_date_str}: {mom_pct}% MoM"
                            )
                            last_val = df.loc[df["date"] == last_date, "value"].iloc[0]
                            new_val = last_val * (1 + (mom_pct / 100.0))

                            new_row = pd.DataFrame(
                                [{"date": ref_date, "value": new_val}]
                            )
                            df = pd.concat([df, new_row], ignore_index=True)

                            # Save back to CSV so we keep the extrapolated history
                            save_df = df.rename(
                                columns={"date": "Date", "value": "INDCPIALLMINMEI"}
                            )
                            save_df["Date"] = save_df["Date"].dt.strftime("%Y-%m-%d")
                            save_df.to_csv(cpi_path, index=False)
                            print(f"  Appended new extrapolated value: {new_val:.3f}")
                        else:
                            print(
                                f"  CPI is up to date (Latest available: {ref_date_str})"
                            )
                        break
    except Exception as e:
        print(f"  Warning: Could not fetch from TE: {e}")

    issues = validate_cpi(df)
    if issues:
        print(f"  Warnings: {issues}")
    return df, issues


def refresh_wpi(api_key=None):
    print("Fetching WPI data...")
    try:
        df = fetch_fred_series(SERIES["wpi"], api_key=api_key)
    except Exception as e:
        print(f"  FRED WPI failed: {e} — trying RBI DBIE fallback")
        df = fetch_rbi_dbie_series("WPI")
        if df is None or df.empty:
            raise
    issues = validate_wpi(df)
    if issues:
        print(f"  Warnings: {issues}")
    return df, issues


def refresh_rate(api_key=None):
    print("Fetching Interest Rate data...")
    df = fetch_fred_series(SERIES["rate"], api_key=api_key)
    issues = validate_rate(df)
    if issues:
        print(f"  Warnings: {issues}")
    return df, issues


def refresh_fx(api_key=None):
    print("Fetching USD/INR data...")
    df = fetch_fred_series(SERIES["fx"], api_key=api_key)
    issues = validate_fx(df)
    if issues:
        print(f"  Warnings: {issues}")
    return df, issues


def refresh_brent(api_key=None):
    print("Fetching Brent Crude data...")
    df = fetch_fred_series(SERIES["brent"], api_key=api_key)
    issues = validate_brent(df)
    if issues:
        print(f"  Warnings: {issues}")
    return df, issues


def merge_and_save(cpi_df, wpi_df, rate_df, fx_df, brent_df):
    cpi_df = cpi_df.rename(columns={"date": "Date", "value": "INDCPIALLMINMEI"})
    wpi_df = wpi_df.rename(columns={"date": "Date", "value": "WPIATT01INM661N"})
    rate_df = rate_df.rename(columns={"date": "Date", "value": "INTDSRINM193N"})
    fx_df = fx_df.rename(columns={"date": "Date", "value": "DEXINUS"})
    brent_df = brent_df.rename(
        columns={"date": "Date", "value": "Average of DCOILBRENTEU"}
    )
    merged = cpi_df[["Date", "INDCPIALLMINMEI"]]
    merged = merged.merge(wpi_df[["Date", "WPIATT01INM661N"]], on="Date", how="outer")
    merged = merged.merge(rate_df[["Date", "INTDSRINM193N"]], on="Date", how="outer")
    merged = merged.merge(fx_df[["Date", "DEXINUS"]], on="Date", how="outer")
    merged = merged.merge(
        brent_df[["Date", "Average of DCOILBRENTEU"]], on="Date", how="outer"
    )
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged = merged.set_index("Date").resample("ME").last().reset_index()
    # ffill removed intentionally to prevent fake flatlines
    merged.to_csv(MERGED_CSV, index=False)
    print(f"Saved merged dataset to {MERGED_CSV}")
    return merged


def log_refresh(results):
    log_entry = {"timestamp": datetime.utcnow().isoformat() + "Z", "sources": results}
    if REFRESH_LOG.exists():
        with open(REFRESH_LOG) as f:
            log_data = json.load(f)
        if not isinstance(log_data, list):
            log_data = [log_data]
        log_data.append(log_entry)
    else:
        log_data = [log_entry]
    with open(REFRESH_LOG, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Logged refresh to {REFRESH_LOG}")


def fetch_all_sources(api_key=None, cadence="all"):
    print("=" * 60)
    print(f"VATICMACRO DATA REFRESH (Cadence: {cadence})")
    print("=" * 60)
    results = {}

    cpi_df = None
    wpi_df = None
    rate_df = None
    fx_df = None
    brent_df = None

    # Helper to load existing data if we skip fetching
    def _load_existing_series(series_name):
        if not MERGED_CSV.exists():
            return None
        try:
            df = pd.read_csv(MERGED_CSV)
            if series_name == "cpi":
                return (
                    df[["Date", "INDCPIALLMINMEI"]]
                    .rename(columns={"Date": "date", "INDCPIALLMINMEI": "value"})
                    .dropna()
                )
            elif series_name == "wpi":
                return (
                    df[["Date", "WPIATT01INM661N"]]
                    .rename(columns={"Date": "date", "WPIATT01INM661N": "value"})
                    .dropna()
                )
            elif series_name == "rate":
                return (
                    df[["Date", "INTDSRINM193N"]]
                    .rename(columns={"Date": "date", "INTDSRINM193N": "value"})
                    .dropna()
                )
            elif series_name == "fx":
                return (
                    df[["Date", "DEXINUS"]]
                    .rename(columns={"Date": "date", "DEXINUS": "value"})
                    .dropna()
                )
            elif series_name == "brent":
                return (
                    df[["Date", "Average of DCOILBRENTEU"]]
                    .rename(
                        columns={"Date": "date", "Average of DCOILBRENTEU": "value"}
                    )
                    .dropna()
                )
        except Exception as e:
            print(f"Error loading existing data for {series_name}: {e}")
        return None

    fetch_monthly = cadence in ["all", "monthly"]
    fetch_daily = cadence in ["all", "daily"]

    # CPI (Monthly)
    if fetch_monthly:
        try:
            cpi_df, cpi_issues = refresh_cpi(api_key=api_key)
            results["cpi"] = {
                "status": "success",
                "rows": len(cpi_df),
                "last_date": cpi_df["date"].max().strftime("%Y-%m-%d"),
                "issues": cpi_issues,
            }
        except Exception as e:
            results["cpi"] = {"status": "error", "message": str(e)}
    else:
        cpi_df = _load_existing_series("cpi")
        results["cpi"] = {
            "status": "skipped",
            "message": "Not scheduled for this cadence",
        }

    # WPI (Monthly)
    if fetch_monthly:
        try:
            wpi_df, wpi_issues = refresh_wpi(api_key=api_key)
            results["wpi"] = {
                "status": "success",
                "rows": len(wpi_df),
                "last_date": wpi_df["date"].max().strftime("%Y-%m-%d"),
                "issues": wpi_issues,
            }
        except Exception as e:
            results["wpi"] = {"status": "error", "message": str(e)}
    else:
        wpi_df = _load_existing_series("wpi")
        results["wpi"] = {"status": "skipped"}

    # Interest Rate (Monthly / MPC specific)
    if fetch_monthly:
        try:
            rate_df, rate_issues = refresh_rate(api_key=api_key)
            results["rate"] = {
                "status": "success",
                "rows": len(rate_df),
                "last_date": rate_df["date"].max().strftime("%Y-%m-%d"),
                "issues": rate_issues,
            }
        except Exception as e:
            results["rate"] = {"status": "error", "message": str(e)}
    else:
        rate_df = _load_existing_series("rate")
        results["rate"] = {"status": "skipped"}

    # USD/INR (Daily)
    if fetch_daily:
        try:
            fx_df, fx_issues = refresh_fx(api_key=api_key)
            results["fx"] = {
                "status": "success",
                "rows": len(fx_df),
                "last_date": fx_df["date"].max().strftime("%Y-%m-%d"),
                "issues": fx_issues,
            }
        except Exception as e:
            results["fx"] = {"status": "error", "message": str(e)}
    else:
        fx_df = _load_existing_series("fx")
        results["fx"] = {"status": "skipped"}

    # Brent Crude (Daily)
    if fetch_daily:
        try:
            brent_df, brent_issues = refresh_brent(api_key=api_key)
            results["brent"] = {
                "status": "success",
                "rows": len(brent_df),
                "last_date": brent_df["date"].max().strftime("%Y-%m-%d"),
                "issues": brent_issues,
            }
        except Exception as e:
            results["brent"] = {"status": "error", "message": str(e)}
    else:
        brent_df = _load_existing_series("brent")
        results["brent"] = {"status": "skipped"}

    # Ensure we have all dataframes before merging
    if all(df is not None for df in [cpi_df, wpi_df, rate_df, fx_df, brent_df]):
        try:
            merged = merge_and_save(cpi_df, wpi_df, rate_df, fx_df, brent_df)
            results["merged"] = {
                "status": "success",
                "rows": len(merged),
                "date_range": f"{merged['Date'].min().strftime('%Y-%m-%d')} to {merged['Date'].max().strftime('%Y-%m-%d')}",
            }
        except Exception as e:
            results["merged"] = {"status": "error", "message": str(e)}
    else:
        results["merged"] = {
            "status": "error",
            "message": "Core sources missing. Cannot merge.",
        }

    log_refresh(results)
    print("=" * 60)
    print("REFRESH COMPLETE")
    print("=" * 60)
    return results


if __name__ == "__main__":
    import sys

    api_key = os.environ.get("FRED_API_KEY")
    cadence = sys.argv[1] if len(sys.argv) > 1 else "all"
    fetch_all_sources(api_key=api_key, cadence=cadence)
