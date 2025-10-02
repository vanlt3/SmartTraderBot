#!/usr/bin/env python3
"""
Test script để kiểm tra việc fetch data cho tất cả symbols
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

from trading_bot import Config, EnhancedDataManager


async def test_symbol_fetch():
    """Test fetch data for all configured symbols"""
    
    print("🚀 Test Symbol Fetch...")
    print("=" * 50)
    
    # Initialize EnhancedDataManager  
    from trading_bot import APIManager
    api_manager = APIManager()
    dm = EnhancedDataManager(api_manager)
    
    # Test each symbol for H1 timeframe (đủ để test compatibility)
    test_results = {}
    
    for symbol in Config.SYMBOLS:
        print(f"\n🔍 Testing {symbol}...")
        try:
            # Test with H1 data (most commonly available)
            df = await dm.fetch_market_data(symbol, "H1", count=50)
            
            if df is not None and not df.empty:
                test_results[symbol] = {
                    'status': 'SUCCESS',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'latest_price': df['close'].iloc[-1] if 'close' in df.columns else 'N/A'
                }
                print(f"✅ {symbol}: {len(df)} candles fetched")
            else:
                test_results[symbol] = {
                    'status': 'FAILED',
                    'reason': 'No data returned'
                }
                print(f"❌ {symbol}: No data")
                
        except Exception as e:
            test_results[symbol] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ {symbol}: Error - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print("=" * 50)
    
    successful_symbols = []
    failed_symbols = []
    
    for symbol, result in test_results.items():
        if result['status'] == 'SUCCESS':
            successful_symbols.append(symbol)
            print(f"✅ {symbol}: OK ({result['rows']} candles)")
        else:
            failed_symbols.append(symbol)
            print(f"❌ {symbol}: {result['status']}")
    
    print(f"\n🎯 Kết quả: {len(successful_symbols)}/{len(Config.SYMBOLS)} symbols thành công")
    
    if successful_symbols:
        print(f"📈 Symbols có thể phân tích: {', '.join(successful_symbols)}")
    
    if failed_symbols:
        print(f"⚠️  Symbols cần fix: {', '.join(failed_symbols)}")
    
    return len(successful_symbols)


if __name__ == "__main__":
    asyncio.run(test_symbol_fetch())