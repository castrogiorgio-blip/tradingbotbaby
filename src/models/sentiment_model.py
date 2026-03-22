"""
Sentiment Model — wraps FinBERT to produce a daily sentiment signal.

This module provides a clean interface for the ensemble to get
sentiment-based predictions. It aggregates news sentiment into
a directional signal: if sentiment is strongly positive → predict UP.

Usage:
    from src.models.sentiment_model import SentimentModel
    model = SentimentModel()
    signal = model.get_signal("AAPL")  # returns probability of UP
"""
import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.data_pipeline.news_fetcher import NewsFetcher


class SentimentModel:
    """
    Sentiment-based directional signal using FinBERT.

    This isn't a traditional ML model — it converts aggregated news
    sentiment into a probability score that the ensemble can use.
    """

    def __init__(self, use_finbert: bool = True):
        self.news_fetcher = NewsFetcher(use_finbert=use_finbert)
        self.use_finbert = use_finbert
        logger.info(f"SentimentModel initialized (FinBERT={'on' if use_finbert else 'off'})")

    def get_signal(self, symbol: str, days: int = 3) -> dict:
        """
        Get a sentiment-based directional signal for a symbol.

        Aggregates recent news sentiment into a single probability.

        Args:
            symbol: Stock ticker
            days: Days of news to consider

        Returns:
            Dict with:
              probability: float (0-1, where >0.5 = bullish)
              sentiment_score: float (-1 to +1)
              news_count: int
              confidence: float (0-1, based on news volume and consensus)
        """
        df = self.news_fetcher.get_news_sentiment(symbol, days=days)

        if df.empty or len(df) == 0:
            logger.info(f"No news for {symbol}, returning neutral signal")
            return {
                "probability": 0.5,
                "sentiment_score": 0.0,
                "news_count": 0,
                "confidence": 0.0,
            }

        # Aggregate sentiment
        avg_sentiment = df["sentiment_numeric"].mean()
        std_sentiment = df["sentiment_numeric"].std() if len(df) > 1 else 1.0
        news_count = len(df)

        # Convert sentiment (-1 to +1) to probability (0 to 1)
        # Using a sigmoid-like transformation
        probability = 1.0 / (1.0 + np.exp(-3 * avg_sentiment))

        # Confidence is based on:
        # 1. Volume of news (more news = more confident)
        # 2. Consensus (low std = more confident)
        volume_conf = min(news_count / 20, 1.0)  # Max confidence at 20+ articles
        consensus_conf = max(0, 1.0 - std_sentiment) if std_sentiment else 0.5
        confidence = 0.6 * volume_conf + 0.4 * consensus_conf

        result = {
            "probability": float(probability),
            "sentiment_score": float(avg_sentiment),
            "news_count": int(news_count),
            "confidence": float(confidence),
        }

        logger.info(
            f"{symbol} sentiment signal — "
            f"prob: {result['probability']:.3f}, "
            f"score: {result['sentiment_score']:.3f}, "
            f"news: {result['news_count']}, "
            f"conf: {result['confidence']:.3f}"
        )
        return result

    def get_batch_signals(self, symbols: list[str], days: int = 3) -> dict:
        """
        Get sentiment signals for multiple symbols.

        Args:
            symbols: List of tickers
            days: Days of news to consider

        Returns:
            Dict mapping symbol → signal dict
        """
        signals = {}
        for symbol in symbols:
            try:
                signals[symbol] = self.get_signal(symbol, days=days)
            except Exception as e:
                logger.error(f"Sentiment signal failed for {symbol}: {e}")
                signals[symbol] = {
                    "probability": 0.5,
                    "sentiment_score": 0.0,
                    "news_count": 0,
                    "confidence": 0.0,
                }
        return signals
