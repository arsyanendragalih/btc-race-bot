#!/usr/bin/env python3
"""
BTC Race 24h Trading Bot - Agent Genesis Entry
Strategy: Triple EMA Cross + RSI Filter + ADX Trend Strength + Dynamic Position Sizing
Input: JSON lines from stdin (1-minute cbBTC bars from Orca Whirlpool)
Output: JSON order messages to stdout
"""

import json
import sys
import math
from collections import deque

# ─── Configuration ───────────────────────────────────────────────────────────
# EMAs
FAST_EMA = 9
MEDIUM_EMA = 21
SLOW_EMA = 55

# RSI period
RSI_PERIOD = 14

# ADX period
ADX_PERIOD = 14
ADX_THRESHOLD = 20  # Minimum trend strength to trade

# Bollinger Bands (for mean-reversion confirmation)
BB_PERIOD = 20
BB_STD = 2.0

# RSI levels
RSI_OVERSOLD = 35
RSI_OVERSOLD_STRONG = 28
RSI_OVERBOUGHT = 65
RSI_OVERBOUGHT_STRONG = 72

# Risk management
MAX_POSITION_PCT = 0.90  # Max 90% of equity in position
MIN_TRADE_PNL_USD = 0.50  # Minimum PnL in USD to consider a trade valid
MAX_DRAWDOWN_PCT = 0.20   # Hard stop at 20% drawdown from peak equity

# Order fees
FEE_BPS = 4.0    # 4 basis points fee
SLIPPAGE_BPS = 2.0  # 2 basis points slippage
TOTAL_COST_BPS = FEE_BPS + SLIPPAGE_BPS  # 6 bps round-trip cost

# ─── State ───────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.bars: deque = deque(maxlen=500)
        self.equity = 1000.0
        self.peak_equity = 1000.0
        self.position = 0.0  # Positive = long BTC, Negative = short BTC
        self.avg_entry = 0.0
        self.cooldown = 0  # Bars to wait after a trade
        self.cooldown_period = 3
        
    def add_bar(self, bar: dict):
        self.bars.append(bar)
    
    @property
    def prices(self) -> list:
        return [b['close'] for b in self.bars]
    
    @property
    def highs(self) -> list:
        return [b['high'] for b in self.bars]
    
    @property
    def lows(self) -> list:
        return [b['low'] for b in self.bars]
    
    @property
    def volumes(self) -> list:
        return [b.get('volume', 0) for b in self.bars]
    
    @property
    def needs_enough_data(self) -> bool:
        return len(self.bars) >= SLOW_EMA + ADX_PERIOD + 50
    
    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def update_peak(self):
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def pnl_from_position(self) -> float:
        if self.position == 0 or self.avg_entry == 0:
            return 0.0
        return self.position * (self.bars[-1]['close'] - self.avg_entry)

    @property
    def total_value(self) -> float:
        return self.equity + self.pnl_from_position()


# ─── Technical Indicators ────────────────────────────────────────────────────
def ema(values: list, period: int) -> list:
    """Calculate Exponential Moving Average."""
    result = [0.0] * len(values)
    if len(values) < period:
        return result
    
    multiplier = 2.0 / (period + 1)
    # Start with SMA
    result[period - 1] = sum(values[:period]) / period
    
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
    
    return result


def rsi(prices: list, period: int = 14) -> list:
    """Calculate Relative Strength Index."""
    result = [0.0] * len(prices)
    if len(prices) < period + 1:
        return result
    
    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(0, diff))
        losses.append(max(0, -diff))
    
    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period - 1):
        result[i + period] = 50.0  # Neutral for first valid RSI
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    for i in range(1, len(gains) - period + 1):
        avg_gain = (avg_gain * (period - 1) + gains[period - 1 + i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[period - 1 + i]) / period
        
        if avg_loss == 0:
            result[period + i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period + i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def atr(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Calculate Average True Range."""
    result = [0.0] * len(closes)
    if len(closes) < 2:
        return result
    
    trs = [0.0] * len(closes)
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs[i] = tr
    
    if len(trs) >= period:
        result[period - 1] = sum(trs[:period]) / period
        for i in range(period, len(trs)):
            result[i] = (result[i - 1] * (period - 1) + trs[i]) / period
    
    return result


def adx(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Calculate Average Directional Index."""
    result = [0.0] * len(closes)
    if len(closes) < period + 1:
        return result
    
    # Calculate +DM, -DM, TR
    dm_plus = [0.0] * len(closes)
    dm_minus = [0.0] * len(closes)
    
    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    atr_vals = atr(highs, lows, closes, period)
    
    if atr_vals[period - 1] == 0:
        return result
    
    # Smoothed +DI, -DI
    smoothed_plus = sum(dm_plus[1:period + 1])
    smoothed_minus = sum(dm_minus[1:period + 1])
    
    for i in range(period, len(closes)):
        smoothed_plus = smoothed_plus * (period - 1) + dm_plus[i]
        smoothed_minus = smoothed_minus * (period - 1) + dm_minus[i]
        
        atr_smooth = atr_vals[i] * period  # Simplified
        
        if atr_smooth > 0:
            di_plus = 100 * smoothed_plus / atr_smooth
            di_minus = 100 * smoothed_minus / atr_smooth
        else:
            di_plus = 50
            di_minus = 50
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
        result[i] = dx
    
    # Smooth ADX
    if len(result) > period:
        sum_adx = sum(result[period:2 * period])
        result[2 * period - 1] = sum_adx / period
        
        for i in range(2 * period, len(closes)):
            result[i] = (result[i - 1] * (period - 1) + result[i]) / period
    
    return result


def bollinger_bands(prices: list, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
    upper = [0.0] * len(prices)
    middle = [0.0] * len(prices)
    lower = [0.0] * len(prices)
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        
        middle[i] = mean
        upper[i] = mean + std_dev * std
        lower[i] = mean - std_dev * std
    
    return upper, middle, lower


# ─── Strategy ────────────────────────────────────────────────────────────────
class TradingBot:
    def __init__(self):
        self.state = State()
        
    def analyze(self) -> dict | None:
        """Run all indicators and return trading signal."""
        if not self.state.needs_enough_data:
            return None
        
        prices = self.state.prices
        highs = self.state.highs
        lows = self.state.lows
        closes = prices  # aliases
        
        # Calculate indicators
        fast_ema = ema(closes, FAST_EMA)
        medium_ema = ema(closes, MEDIUM_EMA)
        slow_ema = ema(closes, SLOW_EMA)
        rsi_vals = rsi(closes, RSI_PERIOD)
        adx_vals = adx(highs, lows, closes, ADX_PERIOD)
        bb_upper, bb_middle, bb_lower = bollinger_bands(closes, BB_PERIOD, BB_STD)
        atr_vals = atr(highs, lows, closes, 14)
        
        # Current values (index -1)
        i = len(closes) - 1
        prev = i - 1
        
        price = closes[i]
        fast_e = fast_ema[i]
        med_e = medium_ema[i]
        slow_e = slow_ema[i]
        rsi_val = rsi_vals[i]
        adx_val = adx_vals[i] if i < len(adx_vals) else 0
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        bb_mid = bb_middle[i]
        atr_val = atr_vals[i] if i < len(atr_vals) else 0
        volume = self.state.volumes[i]
        prev_volume = self.state.volumes[prev] if prev >= 0 else 0
        
        # ─── Signal Strength Scoring ───────────────────────────────────────
        score = 0  # -10 to +10 range
        reasons = []
        
        # 1. EMA Cross signals
        fast_above_med_prev = fast_ema[prev] > medium_ema[prev]
        fast_above_med_curr = fast_e > med_e
        
        fast_above_slow = fast_e > slow_e
        med_above_slow = med_e > slow_e
        
        if fast_above_med_curr and not fast_above_med_prev:
            score += 2
            reasons.append("fast_ema_crossover_up")
        elif not fast_above_med_curr and fast_above_med_prev:
            score -= 2
            reasons.append("fast_ema_crossover_down")
        
        # 2. Trend direction (EMA alignment)
        if fast_e > med_e > slow_e:
            score += 2
            reasons.append("bullish_ema_aligned")
        elif fast_e < med_e < slow_e:
            score -= 2
            reasons.append("bearish_ema_aligned")
        
        # 3. RSI confirmation
        if rsi_val < RSI_OVERSOLD_STRONG:
            score += 3
            reasons.append("rsi_extremely_oversold")
        elif rsi_val < RSI_OVERSOLD:
            score += 1
            reasons.append("rsi_oversold")
        elif rsi_val > RSI_OVERBOUGHT_STRONG:
            score -= 3
            reasons.append("rsi_extremely_overbought")
        elif rsi_val > RSI_OVERBOUGHT:
            score -= 1
            reasons.append("rsi_overbought")
        
        # 4. ADX trend strength filter
        trend_strength = 0  # -1, 0, 1
        if adx_val > ADX_THRESHOLD:
            if fast_e > med_e:
                trend_strength = 1
                reasons.append(f"strong_uptrend_adx={adx_val:.0f}")
            else:
                trend_strength = -1
                reasons.append(f"strong_downtrend_adx={adx_val:.0f}")
        else:
            reasons.append(f"weak_trend_adx={adx_val:.0f}")
        
        # 5. Bollinger Band mean reversion (confluence)
        if price < bb_low and rsi_val < RSI_OVERSOLD:
            score += 2
            reasons.append("bb_breakdown_rsi_oversold")
        elif price > bb_up and rsi_val > RSI_OVERBOUGHT:
            score -= 2
            reasons.append("bb_breakout_rsi_overbought")
        
        # 6. Volume confirmation
        if prev_volume > 0:
            vol_ratio = volume / prev_volume
            if vol_ratio > 1.5 and trend_strength != 0:
                score += trend_strength * 1
                reasons.append(f"volume_spike_{vol_ratio:.1f}x")
        
        # 7. Pullback in trend (high quality entries)
        if trend_strength == 1 and price < bb_mid and rsi_val < 50:
            score += 1
            reasons.append("bullish_pullback_to_bb_mid")
        elif trend_strength == -1 and price > bb_mid and rsi_val > 50:
            score -= 1
            reasons.append("bearish_pullback_to_bb_mid")
        
        # ─── Decision ──────────────────────────────────────────────────────
        if self.state.drawdown_pct >= MAX_DRAWDOWN_PCT:
            # Emergency exit
            if self.state.position > 0:
                return self._create_close_signal(reasons + ["HARD_DRAWDOWN_STOP"])
            return None
        
        # Close existing position if signal reverses
        if self.state.position > 0 and score <= -3:
            return self._create_close_signal(reasons + ["SIGNAL_REVERSAL_LONG"])
        elif self.state.position < 0 and score >= 3:
            return self._create_close_signal(reasons + ["SIGNAL_REVERSAL_SHORT"])
        
        # Open new position
        if score >= 4 and self.state.position == 0:
            return self._create_buy_signal(score, reasons)
        elif score <= -4 and self.state.position == 0:
            return self._create_sell_signal(score, reasons)
        
        # Scale position on strong signals
        if self.state.position > 0 and score >= 6:
            # Add to winning position
            return self._create_buy_signal(score, reasons + ["SCALE_UP_LONG"])
        elif self.state.position < 0 and score <= -6:
            return self._create_sell_signal(score, reasons + ["SCALE_UP_SHORT"])
        
        return None
    
    def _calculate_position_size(self, price: float, direction: str) -> float:
        """Calculate position size based on ATR risk management."""
        if price <= 0:
            return 0.0
        
        # Get ATR for volatility
        highs = self.state.highs
        lows = self.state.lows
        closes = self.state.prices
        atr_vals = atr(highs, lows, closes, 14)
        atr_val = atr_vals[-1] if atr_vals else price * 0.001
        
        # Risk no more than 2% of equity per trade
        risk_usd = self.state.equity * 0.02
        
        # Position size based on ATR (tighter stop = larger position)
        stop_distance = atr_val * 2.0  # 2x ATR stop
        if stop_distance == 0:
            stop_distance = price * 0.005  # 0.5% default
        
        # Calculate position in BTC
        btc_risk = risk_usd / stop_distance if stop_distance > 0 else 0
        max_btc = (self.state.equity * MAX_POSITION_PCT) / price
        
        size = min(btc_risk, max_btc)
        
        # Ensure minimum position (must be worth at least $5)
        min_btc = 5.0 / price
        size = max(size, min_btc)
        
        return size
    
    def _create_buy_signal(self, score: int, reasons: list) -> dict:
        price = self.state.prices[-1]
        size = self._calculate_position_size(price, 'long')
        
        return {
            'action': 'buy',
            'size': round(size, 8),
            'score': score,
            'reasons': reasons,
            'equity': round(self.state.total_value, 2)
        }
    
    def _create_sell_signal(self, score: int, reasons: list) -> dict:
        price = self.state.prices[-1]
        size = abs(self.state.position)  # Sell all for now
        
        return {
            'action': 'sell',
            'size': round(size, 8),
            'score': score,
            'reasons': reasons,
            'equity': round(self.state.total_value, 2)
        }
    
    def _create_close_signal(self, reasons: list) -> dict:
        price = self.state.prices[-1]
        size = abs(self.state.position)
        action = 'sell' if self.state.position > 0 else 'buy'
        
        return {
            'action': action,
            'size': round(size, 8),
            'score': -10 if self.state.position > 0 else 10,
            'reasons': reasons,
            'equity': round(self.state.total_value, 2)
        }
    
    def process_bar(self, bar: dict) -> dict | None:
        """Process a single minute bar and return order if any."""
        self.state.add_bar(bar)
        
        signal = self.analyze()
        if signal is None:
            return None
        
        # Check if we have enough balance
        price = bar['close']
        if signal['action'] == 'buy':
            cost = price * signal['size']
            if cost > self.state.equity * 0.95:
                # Scale down
                signal['size'] = round((self.state.equity * 0.9) / price, 8)
                signal['reasons'].append("BALANCE_CAPPED")
        
        # Update state
        if signal['action'] == 'buy':
            # Calculate PnL of closing previous short position
            if self.state.position < 0:
                pnl = abs(self.state.position) * (self.avg_entry - price)
                self.state.equity += pnl
            # Set entry
            self.state.avg_entry = price
            self.state.position = signal['size']
        elif signal['action'] == 'sell':
            # Calculate PnL of closing long position
            if self.state.position > 0:
                pnl = self.state.position * (price - self.avg_entry)
                self.state.equity += pnl
            # If selling a long, set new short entry
            self.state.avg_entry = price
            self.state.position = -signal['size']
        
        self.state.update_peak()
        
        return signal


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    bot = TradingBot()
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            bar = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Validate required fields
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(k in bar for k in required):
            continue
        
        signal = bot.process_bar(bar)
        
        if signal:
            # Output order JSON
            output = {
                'action': signal['action'],
                'size': signal['size'],
                'price': bar['close'],
                'score': signal['score'],
                'equity': signal['equity']
            }
            print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
