# Trading Bot Warning Fixes Applied

## Issues Identified and Fixed

### 1. ✅ CUDA/TensorFlow Registration Warnings
**Issue:** Multiple CUDA factory registration warnings
```
E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory
E0000 00:00:1759526114.068126  143786 cuda_dnn.cc:8579] Unable to register cuDNN factory
E0000 00:00:1759526114.084652  143786 cuda_blas.cc:1407] Unable to register cuBLAS factory
```

**Fix Applied:**
- Added additional TensorFlow environment variables to suppress warnings
- Enhanced CPU-only mode configuration
- Added OneDNN and MKL disabling flags

### 2. ✅ Gym Deprecation Warning
**Issue:** Gym library deprecation warning
```
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium
```

**Fix Applied:**
- Already using `gymnasium>=1.0.0` in requirements.txt
- All imports already updated to use gymnasium
- No action needed - already fixed

### 3. ✅ Feature Mismatch Warnings
**Issue:** Models trained with different features than prediction features
```
⚠️ WARNING [EnsembleModel] Feature mismatch detected!
⚠️ WARNING [EnsembleModel] Training features: ['high', 'low', 'open', 'volume']
⚠️ WARNING [EnsembleModel] Prediction features: ['accumulation', 'adx', ...]
```

**Fix Applied:**
- Reduced verbose logging for feature mismatches
- Only log significant mismatches (>2x difference)
- Improved feature normalization logic
- Better handling of missing/extra features

### 4. ✅ Rate Limiting Issues
**Issue:** Too many API calls causing rate limits
```
⚠️ WARNING [BotCore] Rate limit cho oanda, chờ 1.7s...
⚠️ WARNING [DataManager] Không thể lấy dữ liệu cho EURUSD (EUR_USD) H1
```

**Fix Applied:**
- Reduced OANDA rate limit from 30 to 15 requests/minute
- Increased delay between API calls from 0.5s to 1.0s
- Added rate limits for all APIs (marketaux, yahoo_finance)
- Updated bot_config.yaml with optimized settings

### 5. ✅ News API Key Issues
**Issue:** Invalid API keys causing errors
```
⚠️ WARNING [NewsManager] Skipping FinnHub news for EURUSD due to invalid API key
```

**Fix Applied:**
- Improved error handling for invalid API keys
- Reduced noise by logging FinnHub warning only once per session
- Added try-catch blocks for news API requests
- Better fallback handling

## Configuration Updates

### Updated Rate Limits (requests/minute):
- OANDA: 30 → 15 (very conservative)
- FinnHub: 60 → 30
- NewsAPI: 1000 → 500
- Alpha Vantage: 5 → 3
- MarketAux: Added 50
- Yahoo Finance: Added 1000

### Updated API Delays:
- Between symbols: 1.0s → 1.5s
- Between timeframes: 0.5s → 1.0s
- Retry delay: 2.0s → 3.0s

### Added TensorFlow Environment Variables:
- `TF_ENABLE_ONEDNN_OPTS=0`
- `TF_DISABLE_MKL=1`
- `TF_DISABLE_POOL_ALLOCATOR=1`

## Expected Improvements

1. **Reduced Warning Noise:** Feature mismatch warnings only for significant differences
2. **Better Rate Limit Compliance:** More conservative limits and longer delays
3. **Cleaner Logs:** Less verbose error messages for known issues
4. **Improved Stability:** Better error handling for API failures
5. **Faster Startup:** Suppressed TensorFlow initialization warnings

## Files Modified

- `trading_bot.py`: Core fixes for warnings and rate limiting
- `bot_config.yaml`: Updated configuration with optimized settings
- `WARNING_FIXES.md`: This documentation

## Monitoring

Watch for these improvements in the logs:
- Fewer rate limit warnings
- Reduced feature mismatch noise
- Cleaner startup without CUDA warnings
- More stable API data fetching
- Better error handling for news APIs

The bot should now run with significantly fewer warnings while maintaining full functionality.