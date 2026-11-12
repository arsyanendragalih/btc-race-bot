# BTC Race 24h Trading Bot

Agent Genesis entry for make.fun BTC Race 24h bounty.

## Strategy

Triple EMA cross (9/21/55) + RSI filter (14) + ADX trend strength + Bollinger Bands mean-reversion confluence.

## Usage

```bash
cat dataset/bars/CBBTC.ndjson.gz | gunzip -c | python3 main.py
```

## Requirements

Python 3.13+. Standard library only. No external dependencies.

## License

MIT
