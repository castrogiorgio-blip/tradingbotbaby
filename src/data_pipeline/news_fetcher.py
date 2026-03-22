"""
News & Sentiment Fetcher V2 — multi-source sentiment with enhanced features.

Sources:
  1. Finnhub company news (primary)
  2. Finnhub market news (general market sentiment)
  3. Alpha Vantage news sentiment (secondary, cross-validates)

Sentiment scoring:
  - FinBERT (when enabled): fine-tuned financial NLP, most accurate
  - VADER fallback: rule-based, faster, works offline
  - Source credibility weighting: WSJ/Reuters weighted more than blogs

Features produced for ML:
  - Daily sentiment mean, std, skew
  - Sentiment momentum (today vs 3-day average)
  - News volume spike detection
  - Source credibility score
  - Headline topic classification (earnings, M&A, regulation, etc.)

Usage:
    from src.data_pipeline.news_fetcher import NewsFetcher
    fetcher = NewsFetcher()
    df = fetcher.get_news_sentiment("AAPL", days=7)
"""
import pandas as pd
import numpy as np
import finnhub
from datetime import datetime, timedelta
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_api_keys, get_all_tickers, DATA_DIR

# FinBERT loaded lazily
_finbert_pipeline = None

# VADER loaded lazily (lightweight fallback)
_vader_analyzer = None


def _get_finbert():
    """Lazy-load FinBERT sentiment pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("Loading FinBERT model (first time may take a minute)...")
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
        logger.info("FinBERT loaded successfully")
    return _finbert_pipeline


def _get_vader():
    """Lazy-load VADER sentiment analyzer (fast fallback)."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except Exception:
            _vader_analyzer = None
    return _vader_analyzer


# Source credibility weights — higher = more trusted
SOURCE_CREDIBILITY = {
    "reuters": 1.0, "bloomberg": 1.0, "wsj": 0.95,
    "cnbc": 0.85, "marketwatch": 0.85, "barrons": 0.85,
    "ft": 0.90, "yahoo": 0.70, "seekingalpha": 0.60,
    "benzinga": 0.65, "investopedia": 0.70, "motleyfool": 0.55,
    "default": 0.50,
}

# Topic keywords for headline classification
TOPIC_KEYWORDS = {
    "earnings": ["earnings", "revenue", "profit", "loss", "eps", "beat", "miss", "guidance",
                  "quarterly", "annual report", "fiscal"],
    "fed_policy": ["fed", "fomc", "rate hike", "rate cut", "powell", "monetary policy",
                    "interest rate", "taper", "quantitative", "inflation target"],
    "macro": ["gdp", "inflation", "cpi", "unemployment", "jobs report", "nonfarm",
              "retail sales", "consumer confidence", "recession"],
    "merger_acquisition": ["acquire", "merger", "buyout", "takeover", "deal", "bid"],
    "regulation": ["sec", "regulation", "antitrust", "lawsuit", "fine", "compliance", "ban"],
    "product": ["launch", "product", "iphone", "chip", "ai", "autonomous", "patent"],
    "analyst": ["upgrade", "downgrade", "price target", "buy rating", "sell rating", "overweight"],
}


class NewsFetcher:
    """Fetches financial news and computes multi-source sentiment scores."""

    def __init__(self, use_finbert: bool = True):
        keys = get_api_keys()
        self.client = finnhub.Client(api_key=keys["finnhub_api_key"])
        self.use_finbert = use_finbert
        self.alpha_vantage_key = keys.get("alpha_vantage_api_key")
        self.data_dir = DATA_DIR / "raw" / "news"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"NewsFetcher V2 initialized (FinBERT={'on' if use_finbert else 'off'})")

    def get_company_news(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch recent news articles for a symbol from Finnhub."""
        end = datetime.now()
        start = end - timedelta(days=days)

        try:
            news = self.client.company_news(
                symbol,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return pd.DataFrame()

        if not news:
            logger.info(f"No news found for {symbol} in last {days} days")
            return pd.DataFrame()

        records = []
        for article in news:
            source = article.get("source", "").lower()
            credibility = SOURCE_CREDIBILITY.get(source, SOURCE_CREDIBILITY["default"])

            records.append({
                "symbol": symbol,
                "datetime": datetime.fromtimestamp(article.get("datetime", 0)),
                "headline": article.get("headline", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "summary": article.get("summary", ""),
                "category": article.get("category", ""),
                "finnhub_sentiment": article.get("sentiment", None),
                "source_credibility": credibility,
            })

        df = pd.DataFrame(records)
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

        # Classify topics
        df["topics"] = df["headline"].apply(self._classify_topics)

        logger.info(f"Fetched {len(df)} news articles for {symbol}")
        return df

    def get_market_news(self, category: str = "general") -> pd.DataFrame:
        """Fetch general market news (market-wide sentiment)."""
        try:
            news = self.client.general_news(category, min_id=0)
        except Exception as e:
            logger.error(f"Failed to fetch market news: {e}")
            return pd.DataFrame()

        if not news:
            return pd.DataFrame()

        records = []
        for article in news[:50]:
            records.append({
                "datetime": datetime.fromtimestamp(article.get("datetime", 0)),
                "headline": article.get("headline", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "summary": article.get("summary", ""),
                "category": category,
            })

        return pd.DataFrame(records)

    def _classify_topics(self, headline: str) -> list:
        """Classify a headline into topic categories."""
        headline_lower = headline.lower()
        matched_topics = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in headline_lower for kw in keywords):
                matched_topics.append(topic)
        return matched_topics

    def score_sentiment(self, texts: list[str]) -> list[dict]:
        """
        Score sentiment using FinBERT (primary) or VADER (fallback).

        Returns list of dicts with: label, score, method
        """
        if not texts:
            return []

        if self.use_finbert:
            try:
                finbert = _get_finbert()
                truncated = [t[:512] if t else "neutral" for t in texts]
                results = finbert(truncated, truncation=True, max_length=512)
                return [{"label": r["label"], "score": r["score"], "method": "finbert"}
                        for r in results]
            except Exception as e:
                logger.warning(f"FinBERT failed, falling back to VADER: {e}")

        # VADER fallback
        vader = _get_vader()
        if vader is not None:
            results = []
            for text in texts:
                scores = vader.polarity_scores(text)
                compound = scores["compound"]
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                results.append({
                    "label": label,
                    "score": abs(compound),
                    "method": "vader",
                })
            return results

        # Ultimate fallback: neutral
        return [{"label": "neutral", "score": 0.5, "method": "none"}] * len(texts)

    def get_news_sentiment(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch news and compute sentiment scores with credibility weighting."""
        df = self.get_company_news(symbol, days=days)
        if df.empty:
            return df

        # Score headlines
        headlines = df["headline"].tolist()
        sentiments = self.score_sentiment(headlines)

        df["sentiment_label"] = [s["label"] for s in sentiments]
        df["sentiment_score"] = [s["score"] for s in sentiments]
        df["sentiment_method"] = [s["method"] for s in sentiments]

        # Convert to numeric: positive=+1, negative=-1, neutral=0
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        df["sentiment_numeric"] = df.apply(
            lambda row: label_map.get(row["sentiment_label"], 0) * row["sentiment_score"],
            axis=1,
        )

        # Apply credibility weighting
        df["weighted_sentiment"] = df["sentiment_numeric"] * df["source_credibility"]

        logger.info(
            f"{symbol} sentiment — "
            f"positive: {(df['sentiment_label'] == 'positive').sum()}, "
            f"negative: {(df['sentiment_label'] == 'negative').sum()}, "
            f"neutral: {(df['sentiment_label'] == 'neutral').sum()}, "
            f"avg weighted: {df['weighted_sentiment'].mean():.3f}"
        )

        return df

    def get_daily_sentiment_summary(
        self,
        symbol: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get daily aggregated sentiment with enhanced features.

        Returns DataFrame with columns:
          news_count, sentiment_mean, sentiment_std,
          sentiment_positive_pct, sentiment_negative_pct,
          weighted_sentiment, sentiment_momentum,
          news_volume_spike, source_credibility_avg,
          topic_earnings, topic_fed_policy, topic_macro, topic_analyst
        """
        df = self.get_news_sentiment(symbol, days=days)
        if df.empty:
            return pd.DataFrame()

        df["date"] = df["datetime"].dt.date

        # Base aggregations
        daily = df.groupby("date").agg(
            news_count=("headline", "count"),
            sentiment_mean=("sentiment_numeric", "mean"),
            sentiment_std=("sentiment_numeric", "std"),
            sentiment_max=("sentiment_numeric", "max"),
            sentiment_min=("sentiment_numeric", "min"),
            weighted_sentiment=("weighted_sentiment", "mean"),
            source_credibility_avg=("source_credibility", "mean"),
            sentiment_positive_pct=(
                "sentiment_label",
                lambda x: (x == "positive").mean(),
            ),
            sentiment_negative_pct=(
                "sentiment_label",
                lambda x: (x == "negative").mean(),
            ),
        ).reset_index()

        # Topic counts per day
        for topic in ["earnings", "fed_policy", "macro", "analyst"]:
            daily[f"topic_{topic}"] = df.groupby("date")["topics"].apply(
                lambda topics_list: sum(1 for topics in topics_list if topic in topics)
            ).values if len(daily) > 0 else 0

        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date").sort_index()
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

        # Sentiment momentum: today's sentiment vs 3-day rolling average
        daily["sentiment_3d_avg"] = daily["sentiment_mean"].rolling(3, min_periods=1).mean()
        daily["sentiment_momentum"] = daily["sentiment_mean"] - daily["sentiment_3d_avg"]

        # News volume spike: is today's news count abnormally high?
        avg_news = daily["news_count"].rolling(7, min_periods=1).mean()
        daily["news_volume_spike"] = (daily["news_count"] > avg_news * 2).astype(int)

        # Sentiment range (high disagreement = uncertainty)
        daily["sentiment_range"] = daily["sentiment_max"] - daily["sentiment_min"]

        return daily


if __name__ == "__main__":
    fetcher = NewsFetcher(use_finbert=False)

    df = fetcher.get_company_news("AAPL", days=3)
    print(f"\nAAPL news (last 3 days): {len(df)} articles")
    if not df.empty:
        print(df[["datetime", "headline", "source", "source_credibility"]].head(5))

    print("\n--- Testing sentiment ---")
    test_headlines = [
        "Apple reports record quarterly revenue, beating expectations",
        "Tech stocks plunge as Fed signals aggressive rate hikes",
        "Markets close mixed as investors digest jobs data",
    ]
    sentiments = fetcher.score_sentiment(test_headlines)
    for headline, sent in zip(test_headlines, sentiments):
        print(f"  [{sent['label']:>8} {sent['score']:.2f} ({sent['method']})] {headline[:60]}")
