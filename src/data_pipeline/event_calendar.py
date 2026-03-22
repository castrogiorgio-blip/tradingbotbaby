"""
Event Calendar — earnings dates, Fed meetings, and economic events.

Provides critical context that affects stock prices:
  1. Earnings dates: stocks move 3-8% on earnings, high IV before
  2. Fed meetings (FOMC): rate decisions move the entire market
  3. Economic releases: CPI, jobs, GDP affect sectors differently

These events are encoded as features for the ML models to learn from.

Usage:
    from src.data_pipeline.event_calendar import EventCalendar
    cal = EventCalendar()
    events_df = cal.get_events("SPY", days=365)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_api_keys, DATA_DIR


class EventCalendar:
    """Fetches and encodes market-moving events as ML features."""

    # Known FOMC meeting dates (2024-2026) — these are public and fixed
    # Source: federalreserve.gov
    FOMC_DATES = [
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
        # 2026
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
    ]

    # Major economic release days (approximate — typically monthly)
    # These create market-wide volatility
    ECON_EVENTS = {
        "cpi": {"day_of_month": [10, 11, 12, 13, 14], "impact": "high"},     # CPI usually 2nd week
        "nfp": {"day_of_month": [1, 2, 3, 4, 5, 6, 7], "impact": "high"},   # Jobs report 1st Friday
        "ppi": {"day_of_month": [11, 12, 13, 14, 15], "impact": "medium"},   # PPI mid-month
        "retail_sales": {"day_of_month": [13, 14, 15, 16, 17], "impact": "medium"},
    }

    def __init__(self):
        self.fomc_dates = pd.to_datetime(self.FOMC_DATES)
        logger.info(f"EventCalendar initialized ({len(self.fomc_dates)} FOMC dates)")

    def get_earnings_dates(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Get earnings dates for a symbol using Finnhub.

        Returns DataFrame with columns: date, earnings_before_market,
        eps_estimate, eps_actual (if available)
        """
        try:
            import finnhub
            keys = get_api_keys()
            client = finnhub.Client(api_key=keys["finnhub_api_key"])

            end = datetime.now()
            start = end - timedelta(days=days)

            earnings = client.company_earnings(symbol, limit=20)
            if not earnings:
                return pd.DataFrame()

            records = []
            for e in earnings:
                period = e.get("period", "")
                if period:
                    records.append({
                        "date": pd.to_datetime(period),
                        "eps_actual": e.get("actual"),
                        "eps_estimate": e.get("estimate"),
                        "surprise": e.get("surprise"),
                        "surprise_pct": e.get("surprisePercent"),
                    })

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
            return df.sort_values("date")

        except Exception as e:
            logger.warning(f"Could not fetch earnings for {symbol}: {e}")
            return pd.DataFrame()

    def encode_events(self, dates: pd.DatetimeIndex, symbol: str = "SPY") -> pd.DataFrame:
        """
        Encode event proximity as features for each trading day.

        Returns DataFrame with columns:
          - days_to_fomc: days until next FOMC meeting (-3 to +3 window)
          - fomc_week: 1 if within FOMC week, 0 otherwise
          - days_to_earnings: days until next earnings (if applicable)
          - earnings_week: 1 if within earnings week
          - econ_event_risk: 0-1 score for economic event proximity
          - high_event_risk: 1 if multiple events converge
          - month_end: 1 if within 3 days of month end (rebalancing flows)
          - quad_witching: 1 if within 3 days of options expiration (3rd Friday)
        """
        n = len(dates)
        features = pd.DataFrame(index=dates)

        # FOMC proximity
        features["days_to_fomc"] = 999
        features["fomc_week"] = 0
        for fomc_date in self.fomc_dates:
            for i, d in enumerate(dates):
                days_diff = (fomc_date - d).days
                if abs(days_diff) < abs(features.iloc[i]["days_to_fomc"]):
                    features.iloc[i, features.columns.get_loc("days_to_fomc")] = days_diff
                if -2 <= days_diff <= 2:
                    features.iloc[i, features.columns.get_loc("fomc_week")] = 1

        # Cap days_to_fomc at reasonable range
        features["days_to_fomc"] = features["days_to_fomc"].clip(-30, 30)

        # Earnings proximity (for individual stocks, not ETFs like SPY)
        features["days_to_earnings"] = 999
        features["earnings_week"] = 0
        if symbol not in ("SPY", "QQQ", "DIA", "IWM"):
            try:
                earnings_df = self.get_earnings_dates(symbol)
                if not earnings_df.empty:
                    for _, row in earnings_df.iterrows():
                        earn_date = row["date"]
                        for i, d in enumerate(dates):
                            days_diff = (earn_date - d).days
                            if abs(days_diff) < abs(features.iloc[i]["days_to_earnings"]):
                                features.iloc[i, features.columns.get_loc("days_to_earnings")] = days_diff
                            if -2 <= days_diff <= 2:
                                features.iloc[i, features.columns.get_loc("earnings_week")] = 1
            except Exception:
                pass

        features["days_to_earnings"] = features["days_to_earnings"].clip(-30, 30)

        # Economic event risk (approximation based on day of month)
        features["econ_event_risk"] = 0.0
        for d_idx, d in enumerate(dates):
            day = d.day
            risk = 0.0
            for event_name, event_info in self.ECON_EVENTS.items():
                if day in event_info["day_of_month"]:
                    risk += 0.5 if event_info["impact"] == "high" else 0.25
            features.iloc[d_idx, features.columns.get_loc("econ_event_risk")] = min(risk, 1.0)

        # Month end rebalancing (last 3 trading days)
        features["month_end"] = 0
        for i, d in enumerate(dates):
            next_month = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
            days_to_month_end = (next_month - d).days - 1
            if 0 <= days_to_month_end <= 3:
                features.iloc[i, features.columns.get_loc("month_end")] = 1

        # Quad witching (3rd Friday of March, June, September, December)
        features["quad_witching"] = 0
        for i, d in enumerate(dates):
            if d.month in (3, 6, 9, 12):
                # Find 3rd Friday
                first_day = d.replace(day=1)
                first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
                third_friday = first_friday + timedelta(weeks=2)
                if abs((d - third_friday).days) <= 2:
                    features.iloc[i, features.columns.get_loc("quad_witching")] = 1

        # High event risk: multiple events converging
        features["high_event_risk"] = (
            (features["fomc_week"] == 1).astype(int) +
            (features["earnings_week"] == 1).astype(int) +
            (features["econ_event_risk"] > 0.3).astype(int) +
            (features["quad_witching"] == 1).astype(int)
        )
        features["high_event_risk"] = (features["high_event_risk"] >= 2).astype(int)

        return features
