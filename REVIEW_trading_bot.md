# trading_bot.py Review Notes

## Security Issues
- Sensitive credentials (multiple API keys and a Discord webhook URL) are hard-coded directly in `Config`. These secrets will be exposed to anyone who can read the source and will leak if the repo is shared. Move them to environment variables or a secure secret manager. (See lines 101-112.)

## Functional Bugs
- `EnhancedDataManager.fetch_market_data` iterates through `Config.ALTERNATIVE_SYMBOLS`, but immediately skips any variant whose name is not a key in `Config.SYMBOL_MAPPING`. Because the first alternatives like `"XAU_USD"` are not keys (they are intended to be returned by `_get_oanda_symbol`), those valid fallbacks are never attempted. As a result, the method rarely tries anything except the original symbol, defeating the fallback logic. (See lines 495-509.)
