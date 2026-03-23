"""
Trading Bot Web Dashboard — Flask application.

A local web app that shows:
  1. Today's predictions and recommended trades
  2. Portfolio sections (Blue Chip, High Risk, Longer Horizon)
  3. Backtest results and performance metrics
  4. Historical trade log

Runs locally at http://127.0.0.1:5000

Usage:
    python3 -m src.web.app
    # or
    python3 run_dashboard.py
"""
import json
import subprocess
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, get_tickers, DATA_DIR

PROJECT_ROOT = Path(__file__).parent.parent.parent

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ── Background job tracking ──────────────────────────────────

_jobs = {}  # job_id -> {"status": "running"/"done"/"error", "output": str, "started": datetime}


def _run_script_background(job_id: str, cmd: list[str], on_success=None):
    """Run a Python script in the background, capture output. Optionally chain another job on success."""
    _jobs[job_id] = {"status": "running", "output": "", "started": datetime.now().isoformat()}
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600
        )
        _jobs[job_id]["output"] = result.stdout + result.stderr
        _jobs[job_id]["status"] = "done" if result.returncode == 0 else "error"
        _jobs[job_id]["returncode"] = result.returncode
    except subprocess.TimeoutExpired:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["output"] = "Timed out after 10 minutes."
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["output"] = str(e)
    _jobs[job_id]["finished"] = datetime.now().isoformat()

    # Chain: if this job succeeded and there's a follow-up, kick it off
    if _jobs[job_id]["status"] == "done" and on_success:
        on_success()


# ── Helpers ──────────────────────────────────────────────────

def load_latest_predictions():
    """Load most recent predictions from disk."""
    pred_dir = DATA_DIR / "predictions"
    if not pred_dir.exists():
        return {}

    predictions = {}
    for f in pred_dir.glob("daily_predictions_*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
                predictions[f.stem] = data
        except Exception:
            continue

    # Also check for latest single file
    latest_file = pred_dir / "latest_predictions.json"
    if latest_file.exists():
        try:
            with open(latest_file) as fh:
                predictions["latest"] = json.load(fh)
        except Exception:
            pass

    return predictions


def load_backtest_results(symbol="SPY", mode=None):
    """Load backtest results for a symbol and optional mode.

    mode: None/'directional', 'options_credit', 'options_debit'
    """
    bt_dir = DATA_DIR / "backtest"
    results = {}
    suffix = "" if not mode or mode == "directional" else f"_{mode}"

    # Load trades
    trades_file = bt_dir / f"trades_{symbol}{suffix}.csv"
    if trades_file.exists():
        trades_df = pd.read_csv(trades_file)
        results["trades"] = trades_df.to_dict(orient="records")
    else:
        results["trades"] = []

    # Load portfolio history
    portfolio_file = bt_dir / f"portfolio_{symbol}{suffix}.csv"
    if portfolio_file.exists():
        portfolio_df = pd.read_csv(portfolio_file)
        results["portfolio"] = portfolio_df.to_dict(orient="records")
    else:
        results["portfolio"] = []

    # Load metrics
    metrics_file = bt_dir / f"metrics_{symbol}{suffix}.txt"
    if metrics_file.exists():
        metrics = {}
        with open(metrics_file) as f:
            for line in f:
                if ":" in line and not line.startswith("="):
                    key, val = line.strip().split(":", 1)
                    metrics[key.strip()] = val.strip()
        results["metrics"] = metrics
    else:
        results["metrics"] = {}

    # Load predictions
    pred_file = bt_dir / f"predictions_{symbol}.csv"
    if pred_file.exists():
        pred_df = pd.read_csv(pred_file)
        results["predictions"] = pred_df.to_dict(orient="records")
    else:
        results["predictions"] = []

    return results


def load_training_results(symbol="SPY"):
    """Load training metrics."""
    results_file = DATA_DIR / "predictions" / f"training_results_{symbol}.txt"
    if results_file.exists():
        with open(results_file) as f:
            return f.read()
    return "No training results available yet."


def get_portfolio_sections():
    """Organize tickers into portfolio sections."""
    tickers = get_tickers()
    return {
        "blue_chip": {
            "name": "Blue Chip Portfolio",
            "description": "Stable, high-quality stocks — 50% allocation",
            "allocation": "50%",
            "tickers": tickers.get("blue_chip", []),
            "color": "#2563eb",
        },
        "high_risk": {
            "name": "High Risk / High Reward",
            "description": "Volatile stocks with bigger moves — 25% allocation",
            "allocation": "25%",
            "tickers": tickers.get("high_risk", []),
            "color": "#dc2626",
        },
        "longer_horizon": {
            "name": "Longer Horizon",
            "description": "Sector ETFs and commodities — 25% allocation",
            "allocation": "25%",
            "tickers": tickers.get("longer_horizon", []),
            "color": "#16a34a",
        },
    }


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html")


def parse_metric(val, default=0):
    """Parse a metric value that might be formatted (e.g. '$1,019.40', '62.0%', '+1.94%')."""
    if val is None:
        return default
    s = str(val).strip().replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    try:
        n = float(s)
        # If it was a percentage string, convert back to decimal
        if "%" in str(val):
            return n / 100
        return n
    except (ValueError, TypeError):
        return default


@app.route("/api/overview")
def api_overview():
    """Dashboard overview data — shows credit spread results (primary strategy)."""
    predictions = load_latest_predictions()

    # Try credit spreads first (primary strategy), fall back to directional
    credit = load_backtest_results("SPY", mode="options_credit")
    directional = load_backtest_results("SPY")

    # Use credit spread metrics if available, otherwise directional
    if credit.get("trades"):
        metrics = credit.get("metrics", {})
        total_trades = len(credit.get("trades", []))
        strategy_label = "credit_spreads"
    else:
        metrics = directional.get("metrics", {})
        total_trades = len(directional.get("trades", []))
        strategy_label = "directional"

    return jsonify({
        "last_updated": datetime.now().isoformat(),
        "portfolio_value": parse_metric(metrics.get("final_portfolio_value"), 1000),
        "total_return": parse_metric(metrics.get("total_return"), 0),
        "total_trades": total_trades,
        "win_rate": parse_metric(metrics.get("win_rate"), 0),
        "sharpe_ratio": parse_metric(metrics.get("sharpe_ratio"), 0),
        "max_drawdown": parse_metric(metrics.get("max_drawdown"), 0),
        "profit_factor": parse_metric(metrics.get("profit_factor"), 0),
        "expectancy": parse_metric(metrics.get("expectancy"), 0),
        "strategy": strategy_label,
        "has_predictions": bool(predictions),
        "has_backtest": bool(credit.get("trades") or directional.get("trades")),
    })


@app.route("/api/portfolios")
def api_portfolios():
    """Portfolio sections with their tickers."""
    return jsonify(get_portfolio_sections())


@app.route("/api/predictions")
def api_predictions():
    """Latest predictions."""
    predictions = load_latest_predictions()
    return jsonify(predictions)


@app.route("/api/backtest/<symbol>")
def api_backtest(symbol):
    """Backtest results for a symbol. ?mode=options_credit for other modes."""
    mode = request.args.get("mode", None)
    return jsonify(load_backtest_results(symbol.upper(), mode=mode))


@app.route("/api/backtest-comparison/<symbol>")
def api_backtest_comparison(symbol):
    """Compare all backtest modes (directional, credit, debit) for a symbol."""
    symbol = symbol.upper()
    modes = ["directional", "options_credit", "options_debit"]
    comparison = {}
    for mode in modes:
        suffix = "" if mode == "directional" else f"_{mode}"
        metrics_file = DATA_DIR / "backtest" / f"metrics_{symbol}{suffix}.txt"
        if metrics_file.exists():
            metrics = {}
            with open(metrics_file) as f:
                for line in f:
                    if ":" in line and not line.startswith("="):
                        key, val = line.strip().split(":", 1)
                        metrics[key.strip()] = val.strip()
            comparison[mode] = metrics
    return jsonify(comparison)


@app.route("/api/portfolio-history/<symbol>")
def api_portfolio_history(symbol):
    """Portfolio value history for charting."""
    bt_dir = DATA_DIR / "backtest"
    portfolio_file = bt_dir / f"portfolio_{symbol.upper()}.csv"

    if portfolio_file.exists():
        df = pd.read_csv(portfolio_file)
        return jsonify({
            "dates": df["date"].tolist(),
            "values": df.get("total_value", df["capital"]).tolist(),
            "capital": df["capital"].tolist(),
        })
    return jsonify({"dates": [], "values": [], "capital": []})


@app.route("/api/trades/<symbol>")
def api_trades(symbol):
    """Trade history for a symbol."""
    bt_dir = DATA_DIR / "backtest"
    trades_file = bt_dir / f"trades_{symbol.upper()}.csv"

    if trades_file.exists():
        df = pd.read_csv(trades_file)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])


@app.route("/api/training-results/<symbol>")
def api_training_results(symbol):
    """Training results text."""
    return jsonify({"text": load_training_results(symbol.upper())})


# ── Action Endpoints ─────────────────────────────────────────

@app.route("/api/run/predictions", methods=["POST"])
def api_run_predictions():
    """Trigger daily predictions in background, then auto-retrain models."""
    if "predictions" in _jobs and _jobs["predictions"].get("status") == "running":
        return jsonify({"error": "Predictions already running"}), 409

    data = request.json or {}
    symbols = data.get("symbols")
    auto_retrain = data.get("auto_retrain", True)  # On by default

    cmd = [sys.executable, "run_daily_v7.py"]
    if symbols:
        cmd += ["--symbols"] + symbols

    # After predictions succeed, auto-kick-off model retraining
    def _chain_retrain():
        if not auto_retrain:
            return
        logger.info("Predictions done — auto-starting model retrain...")
        train_cmd = [sys.executable, "train_models.py", "--symbols", "SPY"]
        _run_script_background("train", train_cmd)

    thread = threading.Thread(
        target=_run_script_background,
        args=("predictions", cmd),
        kwargs={"on_success": _chain_retrain},
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "job_id": "predictions", "auto_retrain": auto_retrain})


@app.route("/api/run/backtest", methods=["POST"])
def api_run_backtest():
    """Trigger backtest in background."""
    if "backtest" in _jobs and _jobs["backtest"].get("status") == "running":
        return jsonify({"error": "Backtest already running"}), 409

    data = request.json or {}
    symbol = data.get("symbol", "SPY")
    capital = data.get("capital", 1000)

    version = data.get("version", "v8")
    script = "run_backtest_v8.py" if version == "v8" else "run_backtest.py"
    cmd = [sys.executable, script, "--symbol", symbol, "--capital", str(capital)]
    if version == "v8" and data.get("skip_wf_xgb", True):
        cmd.append("--skip-wf-xgb")
    thread = threading.Thread(target=_run_script_background, args=("backtest", cmd))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "job_id": "backtest"})


@app.route("/api/run/train", methods=["POST"])
def api_run_train():
    """Trigger model training in background."""
    if "train" in _jobs and _jobs["train"].get("status") == "running":
        return jsonify({"error": "Training already running"}), 409

    data = request.json or {}
    symbols = data.get("symbols", ["SPY"])

    cmd = [sys.executable, "train_models.py", "--symbols"] + symbols
    thread = threading.Thread(target=_run_script_background, args=("train", cmd))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "job_id": "train"})


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    """Check status of a background job."""
    if job_id not in _jobs:
        return jsonify({"status": "not_found"})
    return jsonify(_jobs[job_id])


@app.route("/api/v10/comparison")
def api_v10_comparison():
    """V10 strategy comparison data — all layers + buy & hold."""
    comparison_file = DATA_DIR / "analysis" / "v10_comparison.json"
    if comparison_file.exists():
        with open(comparison_file) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "V10 comparison data not found. Run: python3 run_backtest_v10.py --compare"}), 404


@app.route("/api/v10/trades/<int:layer>")
def api_v10_trades(layer):
    """V10 trade history for a specific layer (1, 2, or 3)."""
    trades_file = DATA_DIR / "backtest" / f"trades_SPY_v10_layer{layer}.csv"
    if trades_file.exists():
        df = pd.read_csv(trades_file)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])


@app.route("/api/v10/audit")
def api_v10_audit():
    """V10 audit results — phase 1 (stats), phase 2 (regime), phase 3 (hedges)."""
    results = {}
    for phase in ["phase1_results", "phase2_results", "phase3_results"]:
        f = DATA_DIR / "analysis" / f"{phase}.json"
        if f.exists():
            with open(f) as fh:
                results[phase.replace("_results", "")] = json.load(fh)

    # Include v8 vs v9 comparison
    v8v9_file = DATA_DIR / "analysis" / "v8_vs_v9_comparison.json"
    if v8v9_file.exists():
        with open(v8v9_file) as fh:
            results["v8_vs_v9"] = json.load(fh)

    return jsonify(results)


@app.route("/api/v10/portfolio-history/<int:layer>")
def api_v10_portfolio_history(layer):
    """Build a synthetic portfolio value series from V10 trade data."""
    trades_file = DATA_DIR / "backtest" / f"trades_SPY_v10_layer{layer}.csv"
    if not trades_file.exists():
        return jsonify({"dates": [], "values": []})

    df = pd.read_csv(trades_file)
    if df.empty:
        return jsonify({"dates": [], "values": []})

    # Build cumulative equity curve from trade P&L
    capital = 10000.0
    dates = []
    values = []
    running = capital
    for _, row in df.iterrows():
        running += float(row.get("pnl_dollar", 0))
        dates.append(str(row.get("exit_date", "")).split(" ")[0])
        values.append(round(running, 2))

    return jsonify({"dates": dates, "values": values, "initial": capital})


@app.route("/api/run/backtest-v10", methods=["POST"])
def api_run_backtest_v10():
    """Trigger V10 backtest in background."""
    if "backtest_v10" in _jobs and _jobs["backtest_v10"].get("status") == "running":
        return jsonify({"error": "V10 backtest already running"}), 409

    cmd = [sys.executable, "run_backtest_v10.py", "--compare"]
    thread = threading.Thread(target=_run_script_background, args=("backtest_v10", cmd))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "job_id": "backtest_v10"})


@app.route("/api/scheduler/status")
def api_scheduler_status():
    """Check if the daily scheduler is configured."""
    scheduler_info = {
        "schedule": "6:00 AM EST daily (weekdays)",
        "configured": False,
        "cron": "0 6 * * 1-5",
    }

    # Check if crontab has our entry
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if "run_daily.py" in result.stdout:
            scheduler_info["configured"] = True
    except Exception:
        pass

    # Check if launchd plist exists (macOS)
    plist_path = Path.home() / "Library/LaunchAgents/com.tradingbot.daily.plist"
    if plist_path.exists():
        scheduler_info["configured"] = True
        scheduler_info["method"] = "launchd"

    return jsonify(scheduler_info)


@app.route("/api/scheduler/enable", methods=["POST"])
def api_scheduler_enable():
    """Set up the daily 6am EST auto-scheduler on macOS via launchd."""
    project_dir = str(PROJECT_ROOT)
    python_path = sys.executable
    plist_path = Path.home() / "Library/LaunchAgents/com.tradingbot.daily.plist"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tradingbot.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{project_dir}/run_daily.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_dir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{project_dir}/data/logs/daily_run.log</string>
    <key>StandardErrorPath</key>
    <string>{project_dir}/data/logs/daily_run_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>"""

    try:
        # Create logs directory
        (PROJECT_ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)

        # Write plist
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(plist_content)

        # Load the job
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)

        if result.returncode == 0:
            return jsonify({"status": "enabled", "message": "Daily scheduler enabled — runs at 6:00 AM every day"})
        else:
            return jsonify({"status": "error", "message": result.stderr}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/scheduler/disable", methods=["POST"])
def api_scheduler_disable():
    """Disable the daily auto-scheduler."""
    plist_path = Path.home() / "Library/LaunchAgents/com.tradingbot.daily.plist"
    try:
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
        return jsonify({"status": "disabled", "message": "Daily scheduler disabled"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def run():
    """Start the dashboard server."""
    settings = get_settings()
    web_config = settings.get("web", {})
    host = web_config.get("host", "127.0.0.1")
    port = web_config.get("port", 5000)
    debug = web_config.get("debug", False)

    logger.info(f"Starting dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
