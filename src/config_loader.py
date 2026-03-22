"""
Configuration loader — reads .env for API keys and settings.yaml for parameters.
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Find project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
TICKERS_PATH = PROJECT_ROOT / "config" / "tickers.yaml"
DATA_DIR = PROJECT_ROOT / "data"

# Load environment variables from .env
load_dotenv(ENV_PATH)


def get_settings() -> dict:
    """Load settings.yaml configuration."""
    with open(SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)


def get_tickers() -> dict:
    """Load tickers.yaml watchlists."""
    with open(TICKERS_PATH, "r") as f:
        return yaml.safe_load(f)


def get_api_keys() -> dict:
    """Load API keys from environment variables."""
    keys = {
        "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY"),
        "alpaca_base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        "fred_api_key": os.getenv("FRED_API_KEY"),
        "finnhub_api_key": os.getenv("FINNHUB_API_KEY"),
        "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
    }
    # Validate that all required keys are present
    missing = [k for k, v in keys.items() if v is None and k != "alpaca_base_url"]
    if missing:
        raise ValueError(f"Missing API keys in .env file: {', '.join(missing)}")
    return keys


def get_all_tickers() -> list:
    """Get a flat list of all unique tickers across all portfolio sections."""
    tickers_config = get_tickers()
    all_tickers = set()
    for section in ["blue_chip", "high_risk", "longer_horizon"]:
        if section in tickers_config:
            all_tickers.update(tickers_config[section].get("tickers", []))
    return sorted(list(all_tickers))


# Quick access
settings = get_settings()
api_keys = get_api_keys()
tickers_config = get_tickers()
