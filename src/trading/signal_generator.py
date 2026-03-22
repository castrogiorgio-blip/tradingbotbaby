"""
Signal Generator — converts ML model outputs into actionable trade recommendations.

Takes the ensemble prediction and generates a complete trade recommendation
including: direction, option type, suggested strike, expiry, stop-loss, take-profit.

Usage:
    from src.trading.signal_generator import SignalGenerator
    generator = SignalGenerator()
    recommendation = generator.generate("SPY", ensemble_result, current_price=580.0)
"""
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings


class SignalGenerator:
    """Converts ensemble predictions into trade recommendations."""

    def __init__(self):
        settings = get_settings()
        trading_config = settings.get("trading", {})
        options_config = trading_config.get("options", {})

        self.max_risk_per_trade = trading_config.get("max_portfolio_risk_per_trade", 0.05)
        self.daily_loss_limit = trading_config.get("daily_loss_limit", 0.05)
        self.max_open_positions = trading_config.get("max_open_positions", 3)

        self.default_dte = options_config.get("default_dte_range", [7, 14])
        self.target_delta = options_config.get("target_delta", 0.40)
        self.stop_loss_pct = options_config.get("stop_loss_pct", 0.30)
        self.take_profit_pct = options_config.get("take_profit_pct", 0.75)

        # Portfolio section configs
        self.portfolios = settings.get("portfolios", {})
        logger.info("SignalGenerator initialized")

    def generate(
        self,
        symbol: str,
        ensemble_result: dict,
        current_price: float,
        portfolio_value: float = 1000.0,
        portfolio_section: str = "high_risk",
    ) -> dict:
        """
        Generate a complete trade recommendation.

        Args:
            symbol: Stock ticker
            ensemble_result: Output from EnsembleModel.predict()
            current_price: Current stock price
            portfolio_value: Total portfolio value
            portfolio_section: "blue_chip", "high_risk", or "longer_horizon"

        Returns:
            Dict with full trade recommendation
        """
        signal = ensemble_result["signal"]
        confidence = ensemble_result["confidence"]
        direction = ensemble_result["direction"]
        probability = ensemble_result["probability"]

        # Get section-specific parameters
        section_config = self.portfolios.get(portfolio_section, {})
        min_confidence = section_config.get("min_confidence", 0.70)
        allocation_pct = section_config.get("allocation_pct", 0.25)

        # Check if confidence meets section threshold
        if confidence < min_confidence:
            return self._hold_recommendation(symbol, confidence, min_confidence, portfolio_section)

        # Calculate position size
        section_capital = portfolio_value * allocation_pct
        max_risk_amount = section_capital * self.max_risk_per_trade

        # Determine DTE based on portfolio section
        if portfolio_section == "longer_horizon":
            dte_range = section_config.get("dte_range", [30, 90])
        else:
            dte_range = self.default_dte

        # Calculate option parameters
        option_type = "CALL" if direction == "UP" else "PUT"

        # Estimate strike price (ATM or slightly OTM)
        if option_type == "CALL":
            strike = round(current_price * 1.01, 2)  # ~1% OTM call
        else:
            strike = round(current_price * 0.99, 2)  # ~1% OTM put

        # Estimate premium (rough: ATR-based or flat estimate)
        # In production, this would query the actual options chain
        estimated_premium = current_price * 0.02  # ~2% of stock price as rough estimate

        # Position size: how many contracts can we afford within risk limit
        contract_cost = estimated_premium * 100  # Options are per 100 shares
        max_contracts = max(1, int(max_risk_amount / contract_cost))

        # Stop loss and take profit on the option premium
        stop_loss_price = estimated_premium * (1 - self.stop_loss_pct)
        take_profit_price = estimated_premium * (1 + self.take_profit_pct)

        # Determine expiry date
        avg_dte = int(np.mean(dte_range))
        expiry_date = datetime.now() + timedelta(days=avg_dte)
        # Adjust to next Friday (options typically expire on Fridays)
        days_until_friday = (4 - expiry_date.weekday()) % 7
        if days_until_friday == 0 and expiry_date.hour > 16:
            days_until_friday = 7
        expiry_date = expiry_date + timedelta(days=days_until_friday)

        recommendation = {
            "symbol": symbol,
            "action": f"BUY_{option_type}",
            "option_type": option_type,
            "direction": direction,
            "strike": strike,
            "expiry": expiry_date.strftime("%Y-%m-%d"),
            "dte": avg_dte,
            "estimated_premium": round(estimated_premium, 2),
            "contracts": max_contracts,
            "estimated_cost": round(contract_cost * max_contracts, 2),
            "stop_loss": round(stop_loss_price, 2),
            "take_profit": round(take_profit_price, 2),
            "confidence": round(confidence, 4),
            "probability": round(probability, 4),
            "portfolio_section": portfolio_section,
            "risk_amount": round(max_risk_amount, 2),
            "model_contributions": ensemble_result.get("model_contributions", {}),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"RECOMMENDATION: {recommendation['action']} {symbol} "
            f"${strike} {option_type} exp {recommendation['expiry']} "
            f"({max_contracts} contracts @ ~${estimated_premium:.2f}) "
            f"[conf={confidence:.2f}]"
        )

        return recommendation

    def _hold_recommendation(
        self, symbol: str, confidence: float, min_confidence: float, section: str
    ) -> dict:
        """Generate a HOLD recommendation when confidence is too low."""
        return {
            "symbol": symbol,
            "action": "HOLD",
            "reason": f"Confidence {confidence:.2f} below {section} threshold {min_confidence:.2f}",
            "confidence": round(confidence, 4),
            "portfolio_section": section,
            "timestamp": datetime.now().isoformat(),
        }

    def classify_symbol(self, symbol: str) -> str:
        """Determine which portfolio section a symbol belongs to."""
        from src.config_loader import get_tickers

        tickers_config = get_tickers()
        for section in ["blue_chip", "high_risk", "longer_horizon"]:
            if symbol in tickers_config.get(section, {}).get("tickers", []):
                return section
        return "high_risk"  # Default

    def generate_daily_recommendations(
        self,
        symbols: list[str],
        ensemble_results: dict,
        current_prices: dict,
        portfolio_value: float = 1000.0,
    ) -> list[dict]:
        """
        Generate recommendations for all symbols.

        Args:
            symbols: List of tickers
            ensemble_results: Dict mapping symbol → ensemble prediction
            current_prices: Dict mapping symbol → current price
            portfolio_value: Total portfolio value

        Returns:
            List of trade recommendations sorted by confidence
        """
        recommendations = []

        for symbol in symbols:
            if symbol not in ensemble_results or symbol not in current_prices:
                continue

            section = self.classify_symbol(symbol)
            rec = self.generate(
                symbol=symbol,
                ensemble_result=ensemble_results[symbol],
                current_price=current_prices[symbol],
                portfolio_value=portfolio_value,
                portfolio_section=section,
            )
            recommendations.append(rec)

        # Sort: actionable trades first (by confidence), then HOLDs
        recommendations.sort(
            key=lambda r: (r["action"] != "HOLD", r.get("confidence", 0)),
            reverse=True,
        )

        # Enforce max open positions
        active_trades = [r for r in recommendations if r["action"] != "HOLD"]
        if len(active_trades) > self.max_open_positions:
            logger.info(
                f"Limiting to {self.max_open_positions} positions "
                f"(had {len(active_trades)} candidates)"
            )
            for r in active_trades[self.max_open_positions:]:
                r["action"] = "HOLD"
                r["reason"] = "Max open positions reached"

        return recommendations
