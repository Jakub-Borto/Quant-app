# just a helper file

def is_bullish(candle) -> bool:
    """Close > open — candle has a bullish body."""
    return candle["close"] > candle["open"]


def is_bearish(candle) -> bool:
    """Close < open — candle has a bearish body."""
    return candle["close"] < candle["open"]
