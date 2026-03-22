"""
Test Script — Run this on your local machine to verify the entire data pipeline.

Usage:
    cd mytradingbot
    pip install -r requirements.txt
    python test_pipeline.py
"""
from src.config_loader import get_api_keys, get_all_tickers

def test_config():
    """Test 1: Verify all API keys load correctly."""
    print("=" * 60)
    print("TEST 1: Configuration & API Keys")
    print("=" * 60)
    keys = get_api_keys()
    for name, value in keys.items():
        status = "OK" if value else "MISSING"
        display = f"{value[:8]}..." if value else "N/A"
        print(f"  [{status}] {name}: {display}")

    tickers = get_all_tickers()
    print(f"\n  Watchlist: {len(tickers)} tickers loaded")
    print(f"  {tickers[:10]}...")
    return True


def test_price_fetcher():
    """Test 2: Verify Alpaca price data fetching."""
    print("\n" + "=" * 60)
    print("TEST 2: Price Fetcher (Alpaca)")
    print("=" * 60)
    from src.data_pipeline.price_fetcher import PriceFetcher

    fetcher = PriceFetcher()
    df = fetcher.get_historical_bars("SPY", days=30)
    print(f"  SPY bars fetched: {len(df)}")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
    print(f"  Columns: {list(df.columns)}")

    # Test latest bars
    latest = fetcher.get_latest_bars(["SPY", "AAPL", "MSFT"])
    print(f"\n  Latest bars for {len(latest)} symbols:")
    for sym, bar in latest.items():
        print(f"    {sym}: ${bar['close']:.2f}")
    return True


def test_indicators():
    """Test 3: Verify technical indicator computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Technical Indicators")
    print("=" * 60)
    from src.data_pipeline.price_fetcher import PriceFetcher
    from src.data_pipeline.indicator_engine import IndicatorEngine

    fetcher = PriceFetcher()
    df = fetcher.get_historical_bars("SPY", days=365)
    engine = IndicatorEngine()
    df_ind = engine.add_all_indicators(df)

    n_new = len(df_ind.columns) - len(df.columns)
    print(f"  Original columns: {len(df.columns)}")
    print(f"  After indicators: {len(df_ind.columns)} (+{n_new} new)")
    print(f"  Sample indicators (latest values):")
    sample_cols = ["rsi_14", "macd", "bb_width", "adx_14", "atr_14", "obv"]
    for col in sample_cols:
        if col in df_ind.columns:
            val = df_ind[col].iloc[-1]
            print(f"    {col}: {val:.4f}")
    return True


def test_economic_data():
    """Test 4: Verify FRED economic data fetching."""
    print("\n" + "=" * 60)
    print("TEST 4: Economic Data (FRED)")
    print("=" * 60)
    from src.data_pipeline.economic_fetcher import EconomicFetcher

    fetcher = EconomicFetcher()
    latest = fetcher.get_latest_values()
    print(f"  Fetched {len(latest)} economic indicators:")
    for name, info in latest.items():
        print(f"    {name}: {info['value']:.2f} (as of {info['date']})")
    return True


def test_news():
    """Test 5: Verify Finnhub news fetching (without FinBERT for speed)."""
    print("\n" + "=" * 60)
    print("TEST 5: News Fetcher (Finnhub)")
    print("=" * 60)
    from src.data_pipeline.news_fetcher import NewsFetcher

    fetcher = NewsFetcher(use_finbert=False)
    df = fetcher.get_company_news("AAPL", days=3)
    print(f"  AAPL news articles (last 3 days): {len(df)}")
    if not df.empty:
        for _, row in df.head(3).iterrows():
            print(f"    [{row['datetime']}] {row['headline'][:70]}...")
    return True


def test_feature_builder():
    """Test 6: Build complete feature set for one symbol."""
    print("\n" + "=" * 60)
    print("TEST 6: Feature Builder (Full Pipeline)")
    print("=" * 60)
    from src.data_pipeline.feature_builder import FeatureBuilder

    builder = FeatureBuilder(use_finbert=False)  # Skip FinBERT for faster test
    df = builder.build_features("SPY", days=365, include_news=False)

    print(f"  Feature set shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Total features: {len(df.columns)}")
    null_pct = df.isnull().mean().mean() * 100
    print(f"  Average null %: {null_pct:.1f}%")

    if "target_direction" in df.columns:
        up = (df["target_direction"] == 1).sum()
        down = (df["target_direction"] == 0).sum()
        print(f"  Target balance: {up} up days / {down} down days ({up/(up+down)*100:.1f}% up)")
    return True


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# TradingBot ML — Data Pipeline Test Suite")
    print("#" * 60)

    tests = [
        ("Config", test_config),
        ("Price Fetcher", test_price_fetcher),
        ("Indicators", test_indicators),
        ("Economic Data", test_economic_data),
        ("News", test_news),
        ("Feature Builder", test_feature_builder),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"\n  ERROR: {e}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        icon = "✓" if result == "PASS" else "✗"
        print(f"  {icon} {name}: {result}")
    print()
