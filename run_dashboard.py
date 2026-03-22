"""
Launch the TradingBot ML web dashboard.

Usage:
    python3 run_dashboard.py
    # Then open http://127.0.0.1:5000 in your browser
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.web.app import run

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TradingBot ML Dashboard")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("="*50 + "\n")
    run()
