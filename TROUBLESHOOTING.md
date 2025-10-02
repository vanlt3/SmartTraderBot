# Trading Bot Troubleshooting Guide

This document addresses common issues encountered when running the AI/ML Trading Bot.

## Common Issues and Solutions

### 1. FinnHub API 401 Authentication Errors

**Symptoms:**
```
⚠️ WARNING [BotCore] API finnhub trả về status 401
```

**Cause:** The FinnHub API key in the configuration is invalid or expired.

**Solution:** 
- Replace `FINNHUB_API_KEY` in the Config class with a valid API key
- Get a free API key from [FinnHub](https://finnhub.io/)
- The code has been temporarily modified to skip FinnHub requests to prevent crashes

### 2. 'text' KeyError in News Fetching

**Symptoms:**
```
❌ ERROR [NewsManager] Lỗi khi fetch news cho BTCUSD: 'text'
```

**Cause:** Different APIs return different data structures for news articles.

**Solution:** 
- Fixed in the code to handle different API response formats
- Added proper error handling for missing 'text' fields
- Now supports MarketAux, NewsAPI, and FinnHub news formats

### 3. Session Closed Errors

**Symptoms:**
```
⚠️ WARNING [BotCore] Session is closed cho marketaux, đang khởi tạo lại...
```

**Cause:** HTTP session management issues with concurrent requests.

**Solution:** 
- Improved session management in APIManager
- Automatic session recreation when closed
- Added better error handling for network issues

### 4. Gym/Gymnasium Compatibility Warning

**Symptoms:**
```
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium
```

**Cause:** Using outdated gym library.

**Solution:** 
- Updated requirements.txt to use `gymnasium>=1.0.0`
- Updated imports in trading_bot.py to use gymnasium

### 5. CUDA/GPU Issues

**Symptoms:**
```
E0000 00:00:1759434811.451325 cuda_dnn.cc:8579] Unable to register cuDNN factory
```

**Cause:** CUDA is disabled or not properly configured.

**Solution:** 
- CUDA is intentionally disabled with `CUDA_VISIBLE_DEVICES = '-1'`
- The bot runs on CPU by design to avoid GPU dependency issues
- This is expected behavior and not an error

## Verified Working APIs

After testing, the following APIs are working correctly:
- ✅ **MarketAux** - Returns valid news data
- ✅ **NewsAPI** - Returns valid news data  
- ❌ **FinnHub** - Returns 401 errors (needs valid API key)

## Performance Notes

- The bot runs approximately 15-30 seconds per trading cycle
- Most delays are due to API rate limiting (normal behavior)
- Session recreation adds ~1-2 seconds overhead (acceptable)

## Configuration Files Updated

1. `requirements.txt` - Updated gymnasium dependency
2. `trading_bot.py` - Fixed news processing logic and session management
3. API key configuration with clear documentation

## Quick Fixes Applied

1. **Disabled FinnHub** temporarily to prevent 401 errors
2. **Fixed news processing** to handle different API response formats
3. **Improved error handling** for missing keys and network issues
4. **Updated dependencies** for NumPy 2.0 compatibility
5. **Added configuration documentation** for API keys

The bot should now run without crashes, though FinnHub news will be skipped until a valid API key is provided.