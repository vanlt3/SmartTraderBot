#!/usr/bin/env python3
"""
Optimized Trading Bot Starter Script
"""
import os
import asyncio
import yaml
import logging
from pathlib import Path

def load_config():
    """Load configuration from YAML file"""
    config_file = Path("bot_config.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}

async def run_bot_with_optimized_settings():
    """Run bot with rate limiting optimizations"""
    
    # Load configuration
    config = load_config()
    
    # Apply environment optimizations
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Trading Bot with Rate Limit Optimizations...")
    print("📋 Applied optimizations:")
    
    rate_limits = config.get('rate_limits', {})
    if rate_limits:
        print(f"   • OANDA rate limit: {rate_limits.get('oanda', 20)} requests/min")
        print(f"   • API delays configured")
    
    print("   • Sequential data fetching enabled")
    print("   • Enhanced error handling")
    print("   • Mock data fallback enabled")
    print("")
    
    # Import and run the bot
    try:
        from trading_bot import TradingBotController
        
        bot = TradingBotController()
        
        # Override rate limits if configured
        if rate_limits:
            if hasattr(bot, 'api_manager') and bot.api_manager is not None:
                bot.api_manager.rate_limit_config.update(rate_limits)
        
        await bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        logging.exception("Bot startup error")

if __name__ == "__main__":
    asyncio.run(run_bot_with_optimized_settings())