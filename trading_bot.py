#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot Giao dịch Tự động AI/ML Tiên tiến
====================================
Bot giao dịch tổng hợp các công nghệ AI/ML với kiến trúc Master-Specialist
Tích hợp Reinforcement Learning, Ensemble Models và Sentiment Analysis
Author: AI Assistant
Version: 2.0.0
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import requests
import sqlite3
import logging
import warnings
import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

# Machine Learning Libraries
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
import optuna

# Deep Learning
import tensorflow as tf

# Force CPU-only mode to avoid CUDA errors
try:
    # Hide GPU devices
    tf.config.set_visible_devices([], 'GPU')
    # Set CPU threading
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all CPU cores
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all CPU cores
    print("⚠️  Running in CPU-only mode (No GPU available)")
except Exception as e:
    print(f"⚠️  TensorFlow configuration warning: {e}")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Reinforcement Learning
import gymnasium as gym
from gymnasium import spaces
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Technical Analysis
import ta
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator

# Online Learning
from river import linear_model, preprocessing, anomaly

# Suppress warnings and CUDA errors
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA devices
os.environ['TF_DISABLE_GPU'] = '1'  # Additional flag to disable GPU

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# ===== CẤU HÌNH API VÀ CONSTANTS =====
class Config:
    """Quản lý cấu hình toàn cục
    
    NOTE: API Keys below may be expired/invalid for demo purposes.
    For production use, replace with valid API keys from respective services.
    """
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = "FK3YQ1IKSC4E1AL5"
    FINNHUB_API_KEY = "YOUR_VALID_FINNHUB_KEY_HERE"  # Invalid - replace with real key
    MARKETAUX_API_KEY = "CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql"
    NEWSAPI_API_KEY = "abd8f43b808f42fdb8d28fb1c429af72"
    EODHD_API_KEY = "68bafd7d44a7f0.25202650"
    OANDA_API_KEY = "814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d"
    GEMINI_API_KEY = "AIzaSyAdrWcXyYHhQb1F8K2P3L4N5M6O7P8Q9R0"  # Replace with actual key
    
    # Endpoints
    OANDA_URL = "https://api-fxtrade.oanda.com/v3"
    DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1419645732218081290/xamfJQdl5kay1wo6w6gxQRrW77d1jpSzKBstQ16Qvb4t5ncGJ3nIHMmm3MQPNT_E-Bt2"
    
    # Trading Symbols
    SYMBOLS = ["XAUUSD", "EURUSD", "NAS100", "BTCUSD"]
    SYMBOL_MAPPING = {
        "NAS100": "NAS100_USD", 
        "EURUSD": "EUR_USD",   # Try EUR_USD first
        "XAUUSD": "XAU_USD",   # OANDA gold symbol 
        "BTCUSD": "BTC_USD"    # OANDA Bitcoin symbol
    }
    # Alternative mappings for different brokers/APIs
    ALTERNATIVE_SYMBOLS = {
        "XAUUSD": ["XAU_USD", "XAUUSD", "GOLD"],
        "BTCUSD": ["BTC_USD", "BTCUSD", "BTC"],
        "EURUSD": ["EUR_USD", "EURUSD", "USDEUR"],  # Try both formats
        "NAS100": ["NAS100_USD", "NAS100", "NAS_USD"]
    }
    
    # Risk Management
    MAX_POSITION_SIZE = 0.02  # 2% per trade
    MAX_CORRELATION_THRESHOLD = 0.7
    STOP_LOSS_MULTIPLIER = 2.0
    TAKE_PROFIT_MULTIPLIER = 3.0
    
    # Database
    DB_PATH = "trading_bot.db"
    
    # Model Parameters
    TIME_FRAMES = ["M15", "H1", "H4", "D1"]
    FEATURE_WINDOW = 50
    PREDICTION_HORIZON = 1

# ===== HỆ THỐNG LOGGING NÂNG CAO =====
class ColorFormatter(logging.Formatter):
    """Formatter với màu sắc cho console logging"""
    
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'
    
    EMOJI_MAP = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        emoji = self.EMOJI_MAP.get(record.levelname, '📝')
        reset_color = self.RESET
        
        # Format thời gian
        fmt_time = datetime.now().strftime('%H:%M:%S')
        
        # Format level với màu và emoji
        colored_level = f"{log_color}{emoji} {record.levelname:<8}{reset_color}"
        
        # Format message
        message = f"{log_color}{record.getMessage()}{reset_color}"
        
        return f"[{fmt_time}] {colored_level} [{record.name}] {message}"

class EnhancedLogManager:
    """Quản lý logging nâng cao với phân tách module"""
    
    def __init__(self):
        self.loggers = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Thiết lập hệ thống logging chính"""
        
        # Tạo root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Console handler với màu sắc
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ColorFormatter())
        root_logger.addHandler(console_handler)
        
        # File handler cho file log chính
        file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(file_handler)
        
        # Tạo các logger chuyên biệt cho từng module
        modules = [
            'BotCore', 'DataManager', 'FeatureEngineer', 'NewsManager',
            'EnsembleModel', 'RLAgent', 'MasterAgent', 'RiskManager', 'Discord'
        ]
        
        for module in modules:
            logger = logging.getLogger(module)
            logger.setLevel(logging.DEBUG)
            self.loggers[module] = logger
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """Lấy logger cho module cụ thể"""
        return self.loggers.get(module_name, logging.getLogger(module_name))
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Tạo báo cáo tóm tắt từ log files"""
        try:
            with open('trading_bot.log', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            error_count = sum(1 for line in lines if 'ERROR' in line)
            warning_count = sum(1 for line in lines if 'WARNING' in line)
            info_count = sum(1 for line in lines if 'INFO' in line)
            
            return {
                'total_entries': len(lines),
                'errors': error_count,
                'warnings': warning_count,
                'info_messages': info_count,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f'Không thể đọc log file: {e}'}

# Khởi tạo hệ thống logging
LOG_MANAGER = EnhancedLogManager()

# ===== QUẢN LÝ API VÀ MONITORING =====
@dataclass
class APIHealth:
    """Trạng thái sức khỏe của API"""
    name: str
    status: str
    response_time: float
    last_check: datetime
    error_count: int = 0
    success_count: int = 0

class APIMonitoringSystem:
    """Theo dõi sức khỏe của các API"""
    
    def __init__(self):
        self.api_health: Dict[str, APIHealth] = {}
        self.logger = LOG_MANAGER.get_logger('BotCore')
    
    def update_api_health(self, api_name: str, status: str, response_time: float):
        """Cập nhật trạng thái sức khỏe API"""
        now = datetime.now()
        
        if api_name not in self.api_health:
            self.api_health[api_name] = APIHealth(api_name, status, response_time, now)
        else:
            health = self.api_health[api_name]
            health.status = status
            health.response_time = response_time
            health.last_check = now
            
            if status == 'healthy':
                health.success_count += 1
            else:
                health.error_count += 1
    
    def get_unhealthy_apis(self) -> List[str]:
        """Lấy danh sách API không khỏe mạnh"""
        unhealthy = []
        for name, health in self.api_health.items():
            if health.error_count > health.success_count or health.response_time > 5.0:
                unhealthy.append(name)
        return unhealthy

class APIManager:
    """Quản lý tất cả API calls với rate limiting và retry logic"""
    
    def __init__(self):
        self.session = None
        self.rate_limits = {}  # API name -> {last_call: time, next_allowed: time}
        self.monitor = APIMonitoringSystem()
        self.logger = LOG_MANAGER.get_logger('BotCore')
        self.retry_count = 3
        self.timeout = 10
        
        # Rate limit settings (requests per minute)
        self.rate_limit_config = {
            'oanda': 30,  # Reduced from 100 to be more conservative
            'finnhub': 60,
            'newsapi': 1000,
            'alpha_vantage': 5
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def close(self):
        """Close the session manually"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Kiểm tra rate limit cho API"""
        if api_name not in self.rate_limits:
            return True
    
        now = time.time()
        limit_info = self.rate_limits[api_name]
        
        # Nếu đã qua thời gian cho phép call tiếp
        if now >= limit_info['next_allowed']:
            return True
        
        return False
    
    def _update_rate_limit(self, api_name: str):
        """Cập nhật thời gian rate limit"""
        now = time.time()
        rate_per_minute = self.rate_limit_config.get(api_name, 60)
        interval = 60.0 / rate_per_minute
        
        if api_name not in self.rate_limits:
            self.rate_limits[api_name] = {'last_call': now, 'next_allowed': now + interval}
        else:
            limit_info = self.rate_limits[api_name]
            limit_info['last_call'] = now
            limit_info['next_allowed'] = now + interval
    
    async def _make_request(self, url: str, headers: Dict = None, 
                          params: Dict = None, api_name: str = 'generic') -> Optional[Dict]:
        """Thực hiện HTTP request với retry logic"""
        
        # Kiểm tra session và tự động khởi tạo lại nếu cần
        if not self.session or self.session.closed:
            self.logger.warning(f"Session is closed cho {api_name}, đang khởi tạo lại...")
            # Properly close existing session before creating new one
            if self.session and not self.session.closed:
                await self.session.close()
            await self.__aenter__()
        
        # Kiểm tra rate limit
        if not self._check_rate_limit(api_name):
            limit_info = self.rate_limits.get(api_name, {})
            wait_time = limit_info.get('next_allowed', 0) - time.time()
            if wait_time > 0:
                self.logger.warning(f"Rate limit cho {api_name}, chờ {wait_time:.1f}s...")
            return None
        
        start_time = time.time()
        
        for attempt in range(self.retry_count):
            try:
                async with self.session.get(url, headers=headers, params=params) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        self._update_rate_limit(api_name)
                        self.monitor.update_api_health(api_name, 'healthy', response_time)
                        
                        data = await response.json()
                        return data
                    else:
                        self.logger.warning(f"API {api_name} trả về status {response.status}")
                        
                if attempt < self.retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retry {attempt + 1}/{self.retry_count} cho {api_name}, chờ {wait_time}s...")
                    await asyncio.sleep(wait_time)
            
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout cho {api_name}, attempt {attempt + 1}")
            except aiohttp.ClientError as e:
                if "Session is closed" in str(e):
                    self.logger.warning(f"Session closed cho {api_name}, attempt {attempt + 1}")
                    # Recreate session for next attempt
                    if self.session:
                        await self.session.close()
                    await self.__aenter__()
                else:
                    self.logger.error(f"Client error cho {api_name}: {e}")
            except Exception as e:
                self.logger.error(f"Lỗi API call {api_name}: {e}")
        
        self.monitor.update_api_health(api_name, 'unhealthy', time.time() - start_time)
        return None

# ===== QUẢN LÝ DỮ LIỆU NÂNG CAO =====
@dataclass
class DataFreshnessInfo:
    """Thông tin độ mới của dữ liệu"""
    symbol: str
    timeframe: str
    last_update: datetime
    is_fresh: bool
    age_minutes: float
    threshold_minutes: float = 30  # Ngưỡng coi dữ liệu là cũ (phút)

class EnhancedDataFreshnessManager:
    """Quản lý độ mới của dữ liệu"""
    
    def __init__(self):
        self.freshness_data: Dict[str, DataFreshnessInfo] = {}
        self.logger = LOG_MANAGER.get_logger('DataManager')
    
    def update_symbol_data(self, symbol: str, timeframe: str, last_update: datetime):
        """Cập nhật thời gian dữ liệu cuối cùng cho symbol"""
        key = f"{symbol}_{timeframe}"
        age_minutes = (datetime.now() - last_update).total_seconds() / 60
        
        self.freshness_data[key] = DataFreshnessInfo(
            symbol=symbol,
            timeframe=timeframe,
            last_update=last_update,
            is_fresh=age_minutes < 30,
            age_minutes=age_minutes,
            threshold_minutes=30
        )
    
    def get_stale_symbols(self) -> List[str]:
        """Lấy danh sách symbols có dữ liệu cũ"""
        stale_symbols = []
        for info in self.freshness_data.values():
            if not info.is_fresh:
                stale_symbols.append(info.symbol)
        return stale_symbols
    
    def refresh_needed(self, symbol: str, timeframe: str) -> bool:
        """Kiểm tra xem có cần làm mới dữ liệu không"""
        key = f"{symbol}_{timeframe}"
        if key not in self.freshness_data:
            return True
        
        info = self.freshness_data[key]
        return not info.is_fresh

class EnhancedDataManager:
    """Quản lý dữ liệu đa khung thời gian với freshness checking"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.freshness_manager = EnhancedDataFreshnessManager()
        self.logger = LOG_MANAGER.get_logger('DataManager')
        self.data_cache = {}  # symbol_timeframe -> DataFrame
    
    def _get_oanda_symbol(self, symbol: str) -> str:
        """Chuyển đổi symbol sang định dạng OANDA"""
        return Config.SYMBOL_MAPPING.get(symbol, symbol)
    
    async def fetch_market_data(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu thị trường từ OANDA API với fallback symbols"""
        
        # Try multiple symbol mappings for better compatibility
        for symbol_variant in Config.ALTERNATIVE_SYMBOLS.get(symbol, [symbol]):
            try:
                # Check if this variant is supported by Oanda
                if symbol_variant not in Config.SYMBOL_MAPPING:
                    continue
                    
                oanda_symbol = self._get_oanda_symbol(symbol_variant)
                url = f"{Config.OANDA_URL}/instruments/{oanda_symbol}/candles"
                
                # Map timeframe to OANDA format
                tf_mapping = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
                params = {
                    'count': min(count, 5000),  # OANDA limit
                    'granularity': tf_mapping[timeframe],
                    'price': 'M'  # Mid only - đơn giản hóa
                }
                
                headers = {
                    'Authorization': f'Bearer {Config.OANDA_API_KEY}',
                    'Content-Type': 'application/json',
                    'Accept-Datetime-Format': 'RFC3339'  # OANDA recommended header
                }
                
                self.logger.info(f"Fetching {symbol} ({oanda_symbol}) {timeframe} data from OANDA...")
                data = await self.api_manager._make_request(url, headers=headers, params=params, api_name='oanda')
                
                if data and 'candles' in data:
                    df = self._process_oanda_data(data['candles'])
                    cache_key = f"{symbol}_{timeframe}"
                    self.data_cache[cache_key] = df
                    
                    # Cập nhật freshness
                    now = datetime.now()
                    self.freshness_manager.update_symbol_data(symbol, timeframe, now)
                    
                    self.logger.info(f"Đã lấy {len(df)} candles cho {symbol} {timeframe}")
                    return df
                elif data and 'errorMessage' in data:
                    self.logger.warning(f"Oanda API error for {symbol_variant} ({oanda_symbol}) {timeframe}: {data['errorMessage']}")
                    # Continue to next variant
                else:
                    self.logger.warning(f"Không thể lấy dữ liệu cho {symbol_variant} ({oanda_symbol}) {timeframe}")
                    # Continue to next variant
                    
            except Exception as e:
                self.logger.warning(f"Lỗi khi fetch {symbol_variant} ({symbol_variant}): {e}")
                # Continue to next variant
        
        # If all variants failed, try alternative APIs
        self.logger.warning(f"OANDA variants cho {symbol} thất bại, thử alternative APIs...")
        return await self._fetch_alternative_api(symbol, timeframe, count)
    
    async def _fetch_alternative_api(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch data from alternative APIs (Alpha Vantage, Yahoo Finance, etc.)"""
        try:
            # Placeholder for alternative data sources
            self.logger.info(f"🔍 Fetching {symbol} via alternative data sources...")
            
            # Try Alpha Vantage first
            if symbol in ["XAUUSD", "BTCUSD"]:
                return await self._fetch_alpha_vantage(symbol, timeframe, count)
            
            # Try Yahoo Finance as fallback
            return await self._fetch_yahoo_finance(symbol, timeframe, count)
            
        except Exception as e:
            self.logger.warning(f"Alternative API fetch cho {symbol} thất bại: {e}")
            return None
    
    async def _fetch_alpha_vantage(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage for crypto/gold"""
        try:
            # Map symbols to Alpha Vantage format
            av_symbol_mapping = {
                "XAUUSD": "XAU",
                "BTCUSD": "BTC"
            }
            
            av_symbol = av_symbol_mapping.get(symbol)
            if not av_symbol:
                return None
            
            # Map timeframe
            tf_mapping = {"D1": "DAILY", "H4": "1_HOUR"}
            if timeframe not in tf_mapping:
                self.logger.warning(f"Timeframe {timeframe} not supported by Alpha Vantage")
                return None
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_' + tf_mapping[timeframe],
                'symbol': av_symbol,
                'apikey': Config.ALPHA_VANTAGE_API_KEY,  # Need to add this to config
                'outputsize': 'compact'  # Last 100 data points
            }
            
            self.logger.info(f"Fetching {symbol} ({av_symbol}) từ Alpha Vantage...")
            data = await self.api_manager._make_request(url, params=params, api_name='alpha_vantage')
            
            # Process Alpha Vantage response format
            if data and 'Time Series (Daily)' in data:
                df = self._process_alpha_vantage_data(data['Time Series (Daily)'])
                if not df.empty:
                    cache_key = f"{symbol}_{timeframe}"
                    self.data_cache[cache_key] = df
                    self.logger.info(f"✅ Đã lấy {len(df)} candles cho {symbol} từ Alpha Vantage")
                    return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Alpha Vantage fetch thất bại cho {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_finance(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance (free fallback)"""
        try:
            # Yahoo Finance doesn't require API key but has rate limits
            self.logger.info(f"Fetching {symbol} từ Yahoo Finance (simulation)...")
            
            # This is a placeholder - would need actual Yahoo Finance integration
            symbols = ["XAUUSD", "EURUSD", "NAS100", "BTCUSD"]
            if symbol in symbols:
                # Create mock data for demonstration
                now = pd.Timestamp.now()
                mock_data = []
                
                for i in range(min(count, 100)):
                    timestamp = now - pd.Timedelta(hours=i)
                    base_price = {"XAUUSD": 2000, "EURUSD": 1.1, "NAS100": 15000, "BTCUSD": 65000}.get(symbol, 100)
                    price_noise = np.random.normal(0, base_price * 0.01)
                    
                    mock_data.append({
                        'timestamp': timestamp,
                        'open': base_price + price_noise,
                        'high': base_price + price_noise + abs(np.random.normal(0, base_price * 0.005)),
                        'low': base_price + price_noise - abs(np.random.normal(0, base_price * 0.005)),
                        'close': base_price + price_noise + np.random.normal(0, base_price * 0.002)
                    })
                
                df = pd.DataFrame(mock_data[::-1])  # Reverse to chronological order
                cache_key = f"{symbol}_{timeframe}"
                self.data_cache[cache_key] = df
                self.logger.info(f"✅ Mock data created cho {symbol}: {len(df)} candles")
                return df
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch thất bại cho {symbol}: {e}")
            return None
    
    def _process_alpha_vantage_data(self, data: dict) -> pd.DataFrame:
        """Process Alpha Vantage time series data"""
        rows = []
        for date, values in data.items():
            rows.append({
                'timestamp': pd.to_datetime(date),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close'])
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def _process_oanda_data(self, candles: List[Dict]) -> pd.DataFrame:
        """Xử lý dữ liệu candles từ OANDA"""
        data = []
        
        for candle in candles:
            if candle['complete']:
                row = {
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    async def get_fresh_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu mới nhất (tự động làm mới nếu cần)"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Kiểm tra cache
        if cache_key in self.data_cache:
            if not self.freshness_manager.refresh_needed(symbol, timeframe):
                return self.data_cache[cache_key]
        
        # Fetch fresh data
        return await self.fetch_market_data(symbol, timeframe)
    
    def get_features(self, symbol: str, timeframe: str = 'H1') -> Optional[np.ndarray]:
        """Extract features for RL Agent from cached data"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            if cache_key not in self.data_cache:
                self.logger.warning(f"No cached data available for {symbol} {timeframe}")
                return None
            
            df = self.data_cache[cache_key]
            
            if df.empty or len(df) < 10:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} candles")
                return None
            
            # Extract basic OHLC features
            features = []
            
            # Recent OHLC data (relative to recent close)
            recent_close = df['close'].iloc[-1]
            
            # Normalize OHLC features
            features.extend([
                df['open'].iloc[-1] / recent_close - 1.0,   # open_norm
                df['high'].iloc[-1] / recent_close - 1.0,   # high_norm  
                df['low'].iloc[-1] / recent_close - 1.0,    # low_norm
                df['close'].iloc[-1] / recent_close - 1.0,  # close_norm (should be 0.0)
            ])
            
            # Volume feature (if available)
            if 'volume' in df.columns:
                volume_norm = df['volume'].iloc[-1] / df['volume'].rolling(10).mean().iloc[-1]
                features.append(volume_norm - 1.0)
            else:
                features.append(0.0)  # Default volume feature
            
            # Price momentum features
            for window in [5, 10]:
                if len(df) >= window:
                    momentum = (df['close'].iloc[-1] / df['close'].iloc[-window] - 1.0)
                    features.append(momentum)
                else:
                    features.append(0.0)
            
            # Volatility features
            if len(df) >= 10:
                returns = df['close'].pct_change().dropna()
                volatility = returns.tail(10).std()
                features.append(volatility)
            else:
                features.append(0.0)
            
            # High-Low spread feature
            price_spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / recent_close
            features.append(price_spread)
            
            # Convert to numpy array and clip extreme values
            feature_array = np.array(features, dtype=np.float32)
            feature_array = np.clip(feature_array, -1.0, 1.0)  # Clip to [-1, 1] range
            
            self.logger.debug(f"Extracted {len(feature_array)} features for {symbol}")
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {symbol}: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> np.ndarray:
        """Return default feature array when extraction fails"""
        return np.array([0.0] * 10, dtype=np.float32)

# ===== ENGINEERING ĐẶC TRƯNG NÂNG CAO =====
class AdvancedFeatureEngineer:
    """Xây dựng đặc trưng kỹ thuật nâng cao và ML"""
    
    def __init__(self):
        self.logger = LOG_MANAGER.get_logger('FeatureEngineer')
    
    def engineer_all_features(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """Tính toán tất cả các đặc trưng cho DataFrame"""
        
        if df.empty:
            self.logger.warning("DataFrame rỗng, không thể tính toán features")
            return df
        
        # Sao chép DataFrame để không thay đổi gốc
        features_df = df.copy()
        
        try:
            # 1. Các chỉ báo cơ bản
            features_df = self._add_basic_indicators(features_df)
            
            # 2. Đặc trưng thống kê
            features_df = self._add_statistical_features(features_df)
            
            # 3. Mô hình nến
            features_df = self._add_candlestick_patterns(features_df)
            
            # 4. Cấu trúc thị trường
            features_df = self._add_market_structure_features(features_df)
            
            # 5. Đặc trưng Wyckoff
            features_df = self._add_wyckoff_features(features_df)
            
            # 6. Supply/Demand zones
            features_df = self._add_supply_demand_features(features_df)
            
            # 7. RSI Divergence
            features_df = self._add_rsi_divergence(features_df)
            
            # 8. Market Regime detection
            features_df = self._add_market_regime(features_df)
            
            self.logger.info(f"Đã tính toán {len(features_df.columns)} features cho {symbol}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán features: {e}")
        
        return features_df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các chỉ báo kỹ thuật cơ bản"""
        
        # ATR (Average True Range)
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        # ADX (Average Directional Index)
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Pivot Points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_1'] = 2 * df['pivot'] - df['high']
        df['resistance_2'] = df['pivot'] + (df['high'] - df['low'])
        df['support_2'] = df['pivot'] - (df['high'] - df['low'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng thống kê"""
        
        # Rolling returns
        for window in [5, 10, 20]:
            df[f'returns_{window}'] = df['close'].pct_change(window)
            df[f'returns_{window}_std'] = df[f'returns_{window}'].rolling(window).std()
        
        # Price volatility
        df['volatility'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
        # Statistical moments
        df['skew'] = df['close'].rolling(20).skew()
        df['kurtosis'] = df['close'].rolling(20).kurt()
        
        # Moving averages
        for ma_period in [5, 10, 20, 50, 200]:
            df[f'sma_{ma_period}'] = df['close'].rolling(ma_period).mean()
            df[f'price_vs_sma_{ma_period}'] = df['close'] / df[f'sma_{ma_period}'] - 1
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các mô hình nến Nhật"""
        
        # Body và Shadow sizes
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Doji pattern
        df['doji'] = (df['body_size'] / (df['high'] - df['low']) < 0.1).astype(int)
        
        # Hammer pattern
        hammer_conditions = (
            (df['lower_shadow'] > 2 * df['body_size']) &
            (df['upper_shadow'] < df['body_size']) &
            (df['close'] > df['open'])
        )
        df['hammer'] = hammer_conditions.astype(int)
        
        # Bullish/Bearish Engulfing
        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        ).astype(int)
        
        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        ).astype(int)
        
        return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm đặc trưng cấu trúc thị trường"""
        
        # Higher highs / Lower lows
        df['hh'] = (df['high'] > df['high'].rolling(5).max().shift(1)).astype(int)
        df['lh'] = (df['high'] < df['high'].rolling(5).max().shift(1)).astype(int)
        df['hl'] = (df['low'] > df['low'].rolling(5).min().shift(1)).astype(int)
        df['ll'] = (df['low'] < df['low'].rolling(5).min().shift(1)).astype(int)
        
        # Swing points
        df['swing_high'] = (df['high'] >= df['high'].rolling(5).max()).astype(int)
        df['swing_low'] = (df['low'] <= df['low'].rolling(5).min()).astype(int)
        
        return df
    
    def _add_wyckoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng Wyckoff"""
        
        # Spring signal (giá phá thấp nhưng đóng lại cao)
        df['spring'] = (
            (df['low'] < df['low'].rolling(20).min().shift(1)) &
            (df['close'] > df['open']) &
            (df['volume'] > df['volume'].rolling(20).mean())
        ).astype(int)
        
        # Upthrust signal (giá phá cao nhưng đóng lại thấp)
        df['upthrust'] = (
            (df['high'] > df['high'].rolling(20).max().shift(1)) &
            (df['close'] < df['open']) &
            (df['volume'] > df['volume'].rolling(20).mean())
        ).astype(int)
        
        # Accumulation/Distribution phases
        # Simplified implementation
        price_momentum = df['close'].diff(10)
        volume_momentum = df['volume'].diff(10)
        
        df['accumulation'] = ((price_momentum < 0) & (volume_momentum > 0)).astype(int)
        df['distribution'] = ((price_momentum > 0) & (volume_momentum > 0)).astype(int)
        
        return df
    
    def _add_supply_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng Supply/Demand zones"""
        
        # Tìm các vùng Supply (kháng cự) và Demand (hỗ trợ)
        lookback = 20
        
        # Supply zones: các đỉnh với rejection
        supply_zones = []
        for i in range(lookback, len(df)):
            window_high = df['high'].iloc[i-lookback:i].max()
            current_high = df['high'].iloc[i]
            
            if current_high == window_high:
                # Kiểm tra rejection từ vùng này
                rejected = (
                    df['close'].iloc[i] < window_high * 0.995 and
                    df['volume'].iloc[i] > df['volume'].iloc[i-lookback:i].mean()
                )
                if rejected:
                    supply_zones.append(i)
        
        # Demand zones: các đáy với bounce
        demand_zones = []
        for i in range(lookback, len(df)):
            window_low = df['low'].iloc[i-lookback:i].min()
            current_low = df['low'].iloc[i]
            
            if current_low == window_low:
                # Kiểm tra bounce từ vùng này
                bounced = (
                    df['close'].iloc[i] > window_low * 1.005 and
                    df['volume'].iloc[i] > df['volume'].iloc[i-lookback:i].mean()
                )
                if bounced:
                    demand_zones.append(i)
        
        # Tính khoảng cách từ giá hiện tại đến các zone gần nhất
        df['distance_to_nearest_supply'] = float('inf')
        df['distance_to_nearest_demand'] = float('inf')
        
        for zone_idx in supply_zones:
            zone_price = df['high'].iloc[zone_idx]
            distance = np.abs(df['close'] - zone_price) / df['close'] * 100
            df['distance_to_nearest_supply'] = np.minimum(
                df['distance_to_nearest_supply'], distance
            )
        
        for zone_idx in demand_zones:
            zone_price = df['low'].iloc[zone_idx]
            distance = np.abs(df['close'] - zone_price) / df['close'] * 100
            df['distance_to_nearest_demand'] = np.minimum(
                df['distance_to_nearest_demand'], distance
            )
        
        return df
    
    def _add_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phát hiện RSI Divergence"""
        
        df['bullish_rsi_div'] = 0
        df['bearish_rsi_div'] = 0
        
        # Tìm các swing highs và lows
        for i in range(5, len(df) - 5):
            # Bullish divergence: giá tạo lower low nhưng RSI tạo higher low
            if (df['low'].iloc[i] < df['low'].iloc[i-5:i].min() and
                df['low'].iloc[i+5:i+10].max() < df['low'].iloc[i] and
                df['rsi'].iloc[i] > df['rsi'].iloc[i-5:i].mean()):
                df['bullish_rsi_div'].iloc[i] = 1
            
            # Bearish divergence: giá tạo higher high nhưng RSI tạo lower high
            if (df['high'].iloc[i] > df['high'].iloc[i-5:i].max() and
                df['high'].iloc[i+5:i+10].min() > df['high'].iloc[i] and
                df['rsi'].iloc[i] < df['rsi'].iloc[i-5:i].mean()):
                df['bearish_rsi_div'].iloc[i] = 1
        
        return df
    
    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phân loại market regime: Trending vs Sideways"""
        
        # Tính ADX để xác định xu hướng
        adx_value = df['adx'].fillna(0)
        
        # Tính slope của moving average để xác định direction
        ma_slope = df['sma_20'].diff(5) / df['sma_20'].shift(5) * 100
        
        # Xác định regime
        conditions = [
            (adx_value > 25) & (ma_slope > 0.1),  # Uptrend
            (adx_value > 25) & (ma_slope < -0.1),  # Downtrend
            adx_value <= 25  # Sideways
        ]
        
        choices = ['uptrend', 'downtrend', 'sideways']
        df['market_regime'] = np.select(conditions, choices, default='sideways')
        
        return df

# ===== QUẢN LÝ TIN TỨC VÀ SENTIMENT ANALYSIS =====
class GeminiSentimentAnalyzer:
    """Phân tích sentiment bằng Gemini AI"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.logger = LOG_MANAGER.get_logger('NewsManager')
        
    async def analyze_sentiment(self, text: str) -> float:
        """Phân tích sentiment của text, trả về score từ -1.0 đến 1.0"""
        
        try:
            prompt = f"""
            Analyze the sentiment of this trading/financial news text and respond with only a number between -1.0 (very negative) and 1.0 (very positive):
            
            Text: {text[:500]}
            
            Consider:
            - Market impact (positive=higher prices, negative=lower prices)
            - Economic implications
            - Investor confidence
            - Company/financial performance
            
            Respond with only the sentiment score number:
            """
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={Config.GEMINI_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 10
                }
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Fallback nếu Gemini không khả dụng
            sentiment_score = self._simple_sentiment_fallback(text)
            return sentiment_score
            
        except Exception as e:
            self.logger.warning(f"Lỗi Gemini sentiment analysis: {e}, dùng fallback")
            return self._simple_sentiment_fallback(text)
    
    def _simple_sentiment_fallback(self, text: str) -> float:
        """Simple sentiment analysis fallback"""
        
        positive_words = [
            'bull', 'bullish', 'rise', 'gain', 'increase', 'up', 'positive', 
            'growth', 'strong', 'robust', 'surge', 'rally', 'optimism',
            'beat', 'exceed', 'outperform', 'better', 'improve'
        ]
        
        negative_words = [
            'bear', 'bearish', 'fall', 'drop', 'decline', 'down', 'negative',
            'weak', 'poor', 'crash', 'plunge', 'pessimistic', 'concern',
            'miss', 'disappoint', 'worse', 'deteriorate'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        pos_ratio = pos_count / total_words
        neg_ratio = neg_count / total_words
        
        score = pos_ratio - neg_ratio
        return np.clip(score * 5, -1.0, 1.0)  # Scale và clamp

class NewsEconomicManager:
    """Quản lý tin tức và lịch kinh tế với sentiment analysis"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.sentiment_analyzer = GeminiSentimentAnalyzer(api_manager)
        self.logger = LOG_MANAGER.get_logger('NewsManager')
        self.news_cache = {}
    
    async def fetch_news_sentiment(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Lấy tin tức và phân tích sentiment"""
        
        news_data = {
            'sentiment_score': 0.0,
            'news_count': 0,
            'positive_news': 0,
            'negative_news': 0,
            'latest_news_items': []
        }
        
        try:
            # Fetch từ nhiều nguồn song song
            tasks = []
            
            # Finnhub News
            if symbol in ['EURUSD', 'NAS100']:  # Forex/Index symbols
                tasks.append(self._fetch_finnhub_news(symbol))
            
            # MarketAux News
            tasks.append(self._fetch_marketaux_news(symbol))
            
            # NewsAPI News
            tasks.append(self._fetch_newsapi_news(symbol))
            
            # Chạy tất cả tasks song song
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_news = []
            for result in results:
                if isinstance(result, list):
                    # Handle MarketAux direct list results
                    for article in result:
                        # MarketAux structure: title, description  
                        title = article.get('title', '')
                        description = article.get('description', '')
                        if title or description:
                            all_news.append({'text': f"{title} {description}".strip()})
                elif isinstance(result, dict) and 'articles' in result:
                    # Handle NewsAPI structure
                    for article in result['articles']:
                        title = article.get('title', '')
                        description = article.get('description', '')
                        if title or description:
                            all_news.append({'text': f"{title} {description}".strip()})
                elif isinstance(result, dict):
                    # Handle FinnHub structure (headline, summary)
                    title = result.get('headline', '')
                    summary = result.get('summary', '')
                    if title or summary:
                        all_news.append({'text': f"{title} {summary}".strip()})
            
            # Phân tích sentiment cho mỗi tin
            if all_news:
                sentiment_scores = []
                for news_item in all_news[:10]:  # Giới hạn 10 tin mới nhất
                    # Safely access text field with error handling
                    text = news_item.get('text', '')
                    if len(text.strip()) > 10:  # Chỉ phân tích nếu có đủ text
                        sentiment = await self.sentiment_analyzer.analyze_sentiment(text)
                        sentiment_scores.append(sentiment)
                        news_data['latest_news_items'].append({
                            'text': text[:100] + '...',
                            'sentiment': sentiment
                        })
                
                if sentiment_scores:
                    # Tính trung bình có trọng số
                    avg_sentiment = np.mean(sentiment_scores)
                    news_data['sentiment_score'] = avg_sentiment
                    news_data['news_count'] = len(sentiment_scores)
                    news_data['positive_news'] = sum(1 for s in sentiment_scores if s > 0)
                    news_data['negative_news'] = sum(1 for s in sentiment_scores if s < 0)
            
            self.logger.info(f"Có {news_data['news_count']} tin tức cho {symbol}, sentiment: {news_data['sentiment_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi fetch news cho {symbol}: {e}")
        
        return news_data
    
    async def _fetch_finnhub_news(self, symbol: str) -> List[Dict]:
        """Fetch news từ Finnhub - Skip FinnHub due to invalid API key for now"""
        # Skip FinnHub due to invalid API key (returns 401)
        # Return empty list to avoid crashes
        self.logger.warning(f"Skipping FinnHub news for {symbol} due to invalid API key")
        return []
    
    async def _fetch_marketaux_news(self, symbol: str) -> List[Dict]:
        """Fetch news từ MarketAux"""
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'symbols': symbol,
            'sort': 'latest',
            'limit': 5,
            'api_token': Config.MARKETAUX_API_KEY
        }
        
        data = await self.api_manager._make_request(url, params=params, api_name='marketaux')
        if data and 'data' in data:
            return data['data']
        return []
    
    async def _fetch_newsapi_news(self, symbol: str) -> Dict:
        """Fetch news từ NewsAPI"""
        # Map symbol to search terms
        search_terms = {
            'XAUUSD': 'gold price',
            'EURUSD': 'euro dollar forex',
            'NAS100': 'NASDAQ nasdaq',
            'BTCUSD': 'bitcoin cryptocurrency'
        }
        
        search_term = search_terms.get(symbol, symbol)
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': search_term,
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'language': 'en',
            'apiKey': Config.NEWSAPI_API_KEY
        }
        
        data = await self.api_manager._make_request(url, params=params, api_name='newsapi')
        return data or {}
    
    async def fetch_economic_calendar(self, hours_ahead: int = 24) -> List[Dict]:
        """Lấy lịch kinh tế"""
        # Simplified economic calendar
        # Trong thực tế sẽ fetch từ các API như ForexFactory, Investing.com
        
        economic_events = [
            {
                'time': datetime.now() + timedelta(hours=2),
                'event': 'Non-Farm Payrolls',
                'importance': 'high',
                'currency': 'USD',
                'forecast': '210K',
                'previous': '185K'
            },
            {
                'time': datetime.now() + timedelta(hours=8),
                'event': 'Consumer Price Index',
                'importance': 'medium',
                'currency': 'EUR',
                'forecast': '2.1%',
                'previous': '1.9%'
            }
        ]
        
        # Filter events trong timeframe
        upcoming_events = [
            event for event in economic_events 
            if event['time'] <= datetime.now() + timedelta(hours=hours_ahead)
        ]
        
        return upcoming_events

# Tiếp tục với các phần tiếp theo...

# ===== ENSEMBLE MODELS VÀ OPTUNA OPTIMIZATION =====
class PurgedGroupTimeSeriesSplit:
    """Custom cross-validation cho time series với gap để tránh data leakage"""
    
    def __init__(self, n_splits=5, gap=10):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """Split data với gap để tránh leakage"""
        n_samples = len(X)
        samples_per_split = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * samples_per_split
            if i == self.n_splits - 1:
                end = n_samples
            else:
                end = (i + 1) * samples_per_split - self.gap
            
            if start >= end:
                continue
                
            train_indices = list(range(start, end))
            
            val_start = end + self.gap
            val_end = val_start + samples_per_split if i < self.n_splits - 1 else n_samples
            
            if val_start < n_samples:
                val_indices = list(range(val_start, val_end))
                yield train_indices, val_indices

class EnsembleModel:
    """Ensemble model với AutoML và hyperparameter optimization"""
    
    def __init__(self):
        self.logger = LOG_MANAGER.get_logger('EnsembleModel')
        self.models = {}
        self.meta_model = None
        self.feature_importance = {}
        self.cv_scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
        self.cv_splitter = PurgedGroupTimeSeriesSplit(n_splits=5, gap=10)
        
        # Models với Optuna optimization
        self.model_configs = {
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': 42
            },
            'lightgbm': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'random_seed': 42,
                'verbose': -1
            },
            'random_forest': {
                'random_state': 42,
                'n_jobs': -1
            }
        }
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               model_name: str) -> Dict[str, Any]:
        """Tối ưu hóa hyperparameters bằng Optuna"""
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
            elif model_name == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                    'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1.0),
                    'max_bin': trial.suggest_int('max_bin', 50, 255),
                    'force_col_wise': True,
                    'verbosity': -1
                }
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
            
            # Train và evaluate model
            cv_scores = []
            for train_idx, val_idx in self.cv_splitter.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                if model_name == 'xgboost':
                    model = xgb.XGBClassifier(**params)
                elif model_name == 'lightgbm':
                    model = lgb.LGBMClassifier(**params)
                elif model_name == 'random_forest':
                    model = RandomForestClassifier(**params)
                
                model.fit(X_train_fold, y_train_fold)
                pred = model.predict(X_val_fold)
                score = accuracy_score(y_val_fold, pred)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        self.logger.info(f"Best {model_name} params: {study.best_params}")
        return study.best_params
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Huấn luyện ensemble models với optimized parameters"""
        
        # Tối ưu hóa từng model
        optimized_params = {}
        
        for model_name in ['xgboost', 'lightgbm', 'random_forest']:
            try:
                params = self.optimize_hyperparameters(X_train, y_train, model_name)
                optimized_params[model_name] = params
                self.logger.info(f"Hoàn thành optimization cho {model_name}")
            except Exception as e:
                self.logger.error(f"Lỗi optimization {model_name}: {e}")
                optimized_params[model_name] = self.model_configs[model_name]
        
        # Train models với optimized parameters
        base_predictions = np.zeros((len(X_train), len(optimized_params)))
        
        for i, (model_name, params) in enumerate(optimized_params.items()):
            try:
                if model_name == 'xgboost':
                    model = xgb.XGBClassifier(**params)
                elif model_name == 'lightgbm':
                    model = lgb.LGBMClassifier(**params)
                elif model_name == 'random_forest':
                    model = RandomForestClassifier(**params)
                
                # Calibrate probabilities with error handling
                try:
                    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                    calibrated_model.fit(X_train, y_train)
                    
                    # Verify calibration was successful
                    if not hasattr(calibrated_model, 'estimator_') or calibrated_model.estimator_ is None:
                        raise ValueError("Calibration failed - no estimator found")
                    
                    self.logger.info(f"Sử dụng calibrated model cho {model_name}")
                except Exception as calib_error:
                    self.logger.warning(f"Calibration failed for {model_name}: {calib_error}")
                    self.logger.info(f"Sử dụng model gốc cho {model_name}")
                    # Fallback to original model
                    calibrated_model = model
                    calibrated_model.fit(X_train, y_train)
                
                self.models[model_name] = calibrated_model
                
                # Get predictions for meta-model
                if hasattr(calibrated_model, 'predict_proba'):
                    predictions = calibrated_model.predict_proba(X_train)[:, 1]
                else:
                    predictions = calibrated_model.decision_function(X_train)
                    # Convert to probabilities
                    scaler = MinMaxScaler()
                    predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
                
                base_predictions[:, i] = predictions
                
                # Calculate feature importance
                estimator_to_check = None
                if hasattr(calibrated_model, 'estimator_') and calibrated_model.estimator_ is not None:
                    estimator_to_check = calibrated_model.estimator_
                else:
                    estimator_to_check = calibrated_model
                
                if hasattr(estimator_to_check, 'feature_importances_'):
                    self.feature_importance[model_name] = estimator_to_check.feature_importances_
                
                self.logger.info(f"Hoàn thành training {model_name}")
                
            except Exception as e:
                self.logger.error(f"Lỗi training {model_name}: {e}")
        
        # Train meta-model (Logistic Regression)
        self.meta_model = LogisticRegression()
        self.meta_model.fit(base_predictions, y_train)
        
        self.logger.info("Hoàn thành ensemble training")
        return self._evaluate_performance(X_train, y_train)
    
    def _evaluate_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Đánh giá performance của ensemble"""
        
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, val_idx in self.cv_splitter.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train ensemble on fold
            temp_ensemble = EnsembleModel()
            temp_ensemble.models = {}
            
            base_predictions = np.zeros((len(X_train_fold), len(self.models)))
            
            for i, (name, model) in enumerate(self.models.items()):
                try:
                    # Create a fresh instance of the model for CV
                    if name == 'xgboost':
                        fresh_model = xgb.XGBClassifier(**self.model_configs[name])
                    elif name == 'lightgbm':
                        fresh_model = lgb.LGBMClassifier(**self.model_configs[name])
                    elif name == 'random_forest':
                        fresh_model = RandomForestClassifier(**self.model_configs[name])
                    
                    fresh_model.fit(X_train_fold, y_train_fold)
                    temp_ensemble.models[name] = fresh_model
                    
                    if hasattr(fresh_model, 'predict_proba'):
                        base_predictions[:, i] = fresh_model.predict_proba(X_train_fold)[:, 1]
                    else:
                        predictions = fresh_model.decision_function(X_train_fold)
                        scaler = MinMaxScaler()
                        predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
                        base_predictions[:, i] = predictions
                except Exception as e:
                    self.logger.error(f"Error training {name} in CV fold: {e}")
                    base_predictions[:, i] = 0.5  # Default probability
            
            temp_meta = LogisticRegression()
            temp_meta.fit(base_predictions, y_train_fold)
            
            #Predict on validation set
            val_base_predictions = np.zeros((len(X_val_fold), len(temp_ensemble.models)))
            for i, (name, model) in enumerate(temp_ensemble.models.items()):
                try:
                    if hasattr(model, 'predict_proba'):
                        val_base_predictions[:, i] = model.predict_proba(X_val_fold)[:, 1]
                    else:
                        predictions = model.decision_function(X_val_fold)
                        scaler = MinMaxScaler()
                        predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
                        val_base_predictions[:, i] = predictions
                except Exception as e:
                    self.logger.error(f"Error predicting with {name} in CV validation: {e}")
                    val_base_predictions[:, i] = 0.5  # Default probability
            
            val_predictions = temp_meta.predict(val_base_predictions)
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, val_predictions))
            cv_scores['precision'].append(precision_score(y_val_fold, val_predictions, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, val_predictions, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, val_predictions, zero_division=0))
        
        # Calculate mean scores
        mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        
        self.logger.info(f"Ensemble CV scores - Accuracy: {mean_scores['accuracy']:.3f}, "
                        f"F1: {mean_scores['f1']:.3f}")
        
        return mean_scores
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán với ensemble"""
        
        # Base model predictions
        base_predictions = np.zeros((len(X), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            try:
                if hasattr(model, 'predict_proba'):
                    base_predictions[:, i] = model.predict_proba(X)[:, 1]
                else:
                    # Fallback to decision function if predict_proba is not available
                    predictions = model.decision_function(X)
                    scaler = MinMaxScaler()
                    predictions = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
                    base_predictions[:, i] = predictions
            except Exception as e:
                self.logger.error(f"Error predicting with {name}: {e}")
                # Use average of other models as fallback
                if i > 0:
                    base_predictions[:, i] = np.mean(base_predictions[:, :i], axis=1)
                else:
                    base_predictions[:, i] = 0.5  # Default probability
        
        # Meta-model prediction
        meta_predictions = self.meta_model.predict_proba(base_predictions)
        
        return np.argmax(meta_predictions, axis=1), meta_predictions[:, 1]

# ===== LSTM MODEL VỚI ATTENTION =====
class LSTMModel:
    """LSTM Model với Attention mechanism"""
    
    def __init__(self, sequence_length: int = 50, feature_dim: int = None):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim  # Will be determined from data
        self.model = None
        self.logger = LOG_MANAGER.get_logger('EnsembleModel')
        self.is_trained = False
        self.scaler = None
    
    def _build_model(self):
        """Xây dựng kiến trúc LSTM với Attention"""
        
        if self.feature_dim is None:
            raise ValueError("Feature dimension must be set before building model")
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.feature_dim), name='input_sequences')
        
        # Batch normalization
        bn_input = BatchNormalization(axis=-1)(inputs)
        
        # First LSTM layer with dropout
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(bn_input)
        lstm1_bn = BatchNormalization(axis=-1)(lstm1)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=128,
            dropout=0.1
        )(lstm1_bn, lstm1_bn)
        
        # Second LSTM layer
        lstm2 = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(attention)
        lstm2_bn = BatchNormalization(axis=-1)(lstm2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm2_bn)
        dense1_drop = Dropout(0.3)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1_drop)
        dense2_drop = Dropout(0.2)(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='prediction')(dense2_drop)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=output)
        
        # Compile model
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.logger.info("Đã xây dựng kiến trúc LSTM với Attention")
    
    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị data thành sequences cho LSTM"""
        
        # Determine feature dimension from data if not set
        if self.feature_dim is None:
            self.feature_dim = X.shape[1]
            self.logger.info(f"LSTM feature dimension set to {self.feature_dim}")
        
        # Scale features
        scaler = sklearn.preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        # Build model with correct feature dimension
        if self.model is None:
            self._build_model()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            if i < len(y):  # Check bounds to prevent index error
                y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2, epochs: int = 100) -> Dict[str, float]:
        """Huấn luyện LSTM model"""
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Validate sequences
        if len(X_seq) == 0:
            self.logger.error("No sequences created from input data")
            return {'error': 'Insufficient data for training'}
        
        if X_seq.shape[1] != self.sequence_length:
            self.logger.error(f"Sequence length mismatch: expected {self.sequence_length}, got {X_seq.shape[1]}")
            return {'error': 'Data preprocessing error'}
        
        # Split data
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
        y_train_seq, y_val_seq = y_seq[:split_idx], y_seq[split_idx:]
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec = self.model.evaluate(X_val_seq, y_val_seq, verbose=0)
        
        self.logger.info(f"LSTM Training completed - Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.3f}")
        
        return {
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'val_precision': val_prec,
            'val_recall': val_rec
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Dự đoán với LSTM model"""
        
        if not self.is_trained:
            self.logger.warning("LSTM model chưa được huấn luyện")
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Prepare sequences
        X_scaled = self.scaler.transform(X)
        X_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        if not X_sequences:
            return np.zeros(len(X)), np.zeros(len(X))
        
        X_seq = np.array(X_sequences)
        
        # Predict
        probabilities = self.model.predict(X_seq, verbose=0).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        # Pad với zeros cho các điểm đầu
        full_predictions = np.zeros(len(X))
        full_probabilities = np.zeros(len(X))
        
        full_predictions[self.sequence_length:] = predictions
        full_probabilities[self.sequence_length:] = probabilities
        
        return full_predictions, full_probabilities

# ===== REINFORCEMENT LEARNING SYSTEM =====
class PortfolioEnvironment(gym.Env):
    """Environment cho portfolio management với RL"""
    
    def __init__(self, data_manager: EnhancedDataManager, feature_engineer: AdvancedFeatureEngineer):
        super(PortfolioEnvironment, self).__init__()
        
        self.data_manager = data_manager
        self.feature_engineer = feature_engineer
        self.logger = LOG_MANAGER.get_logger('RLAgent')
        
        # Symbols
        self.symbols = Config.SYMBOLS
        self.n_symbols = len(self.symbols)
        
        # Action space: allocation weights cho mỗi symbol (-1 to 1), tổng về 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_symbols,), dtype=np.float32)
        
        # Observation space: market state cho mỗi symbol
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_symbols * 30,),  # 30 features per symbol
            dtype=np.float32
        )
        
        # Portfolio state
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = Config.FEATURE_WINDOW
        self.portfolio_value = 100000  # Starting value $100k
        self.cash = 100000
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.trade_history = []
        self.daily_returns = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Lấy observation state"""
        observation = []
        
        for symbol in self.symbols:
            # Simplified market state - trong thực tế sẽ fetch real data
            market_features = np.random.randn(30)  # Random features
            observation.extend(market_features)
        
        return np.array(observation, dtype=np.float32)
    
    def step(self, action: np.ndarray):
        """Execute action và return (observation, reward, done, info)"""
        
        # Normalize action để tổng bình phương = 1
        action = action / (np.linalg.norm(action) + 1e-8)
        
        # Execute trades
        self._execute_trades(action)
        
        # Calculate returns
        portfolio_return = self._calculate_portfolio_return(action)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return, action)
        
        # Update state
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= 1000  # Max 1000 steps
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'daily_return': portfolio_return,
            'positions': self.positions.copy(),
            'action': action.tolist()
        }
        
        return obs, reward, done, info
    
    def _execute_trades(self, action: np.ndarray):
        """Execute trading action"""
        for i, symbol in enumerate(self.symbols):
            target_weight = action[i]
            current_weight = self.positions[symbol] * self._get_symbol_price(symbol) / self.portfolio_value
            
            weight_change = target_weight - current_weight
            
            if abs(weight_change) > 0.01:  # Minimum trade threshold
                self.positions[symbol] += weight_change * self.portfolio_value / self._get_symbol_price(symbol)
                self.trade_history.append({
                    'symbol': symbol,
                    'action': weight_change,
                    'timestamp': self.current_step
                })
    
    def _calculate_portfolio_return(self, action: np.ndarray) -> float:
        """Tính portfolio return"""
        # Simplified - trong thực tế sẽ tính từ price movements
        portfolio_return = np.mean(np.random.randn(self.n_symbols)) * 0.01
        self.daily_returns.append(portfolio_return)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        return portfolio_return
    
    def _calculate_reward(self, portfolio_return: float, action: np.ndarray) -> float:
        """Tính reward function dựa trên Sharpe ratio và các yếu tố khác"""
        
        # Base reward: Sharpe ratio
        if len(self.daily_returns) >= 10:
            sharpe_ratio = np.mean(self.daily_returns) / (np.std(self.daily_returns) + 1e-8)
            sharpe_reward = sharpe_ratio * 100
        else:
            sharpe_reward = portfolio_return * 100
        
        # Penalty for extreme positions
        position_penalty = -np.sum(np.abs(action)) * 0.1
        
        # Penalty for frequent trading
        trading_penalty = -len([t for t in self.trade_history if t['timestamp'] == self.current_step]) * 0.05
        
        total_reward = sharpe_reward + position_penalty + trading_penalty
        
        return total_reward
    
    def _get_symbol_price(self, symbol: str) -> float:
        """Lấy giá symbol hiện tại"""
        # Simplified - return random price
        base_prices = {'XAUUSD': 2000, 'EURUSD': 1.1, 'NAS100': 15000, 'BTCUSD': 40000}
        base_price = base_prices.get(symbol, 1.0)
        noise = np.random.randn() * base_price * 0.01
        return base_price + noise
    
    def render(self):
        """Render environment state"""
        self.logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        self.logger.info(f"Positions: {self.positions}")
        self.logger.info(f"Daily Returns (last 5): {self.daily_returns[-5:]}")

class TrainingCallback(BaseCallback):
    """Custom callback cho RL training"""
    
    def __init__(self, check_freq: int = 10000, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self._logger = LOG_MANAGER.get_logger('RLAgent')
    
    def _on_step(self) -> bool:
        """Called for each step"""
        
        if self.n_calls % self.check_freq == 0:
            # Log training progress
            if hasattr(self.model, 'logger'):
                self._logger.info(f"RL Training step {self.n_calls}")
        
        # Early stopping check
        if hasattr(self.training_env, 'get_attr'):
            portfolio_values = self.training_env.get_attr('portfolio_values', [0])
            if len(portfolio_values) > 0:
                current_value = portfolio_values[0][-1] if portfolio_values[0] else 100000
                
                # Stop if portfolio drops below 70% of starting value
                if current_value < 70000:
                    self._logger.warning("Portfolio value quá thấp, dừng training")
                    return False
        
        return True

class TrendSpecialistAgent:
    """Specialist Agent cho trend analysis và market regime detection"""
    
    def __init__(self):
        self.logger = LOG_MANAGER.get_logger('TrendSpecialistAgent')
        self.trend_signals = {}
        self.market_regime = None
        
    async def analyze_trend(self, symbol: str, data: dict) -> dict:
        """Phân tích trend cho symbol cụ thể"""
        try:
            if symbol not in data or data[symbol].empty:
                return {'trend': 'UNKNOWN', 'confidence': 0.0, 'strength': 0.0}
            
            df = data[symbol].copy()
            
            # Technical indicators cho trend analysis
            if len(df) >= 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                
                # Trend determination
                current_price = df['close'].iloc[-1]
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                
                if current_price > sma_20 > sma_50:
                    trend = 'UPTREND'
                    confidence = 0.8
                elif current_price < sma_20 < sma_50:
                    trend = 'DOWNTREND'
                    confidence = 0.8
                elif sma_20 > sma_50:
                    trend = 'UPTREND'
                    confidence = 0.6
                elif sma_20 < sma_50:
                    trend = 'DOWNTREND'
                    confidence = 0.6
                else:
                    trend = 'SIDEWAYS'
                    confidence = 0.4
                
                # Strength calculation
                price_variance = df['close'].rolling(window=20).std().iloc[-1]
                strength = min(1.0, price_variance / current_price * 100)
                
            else:
                trend = 'INSUFFICIENT_DATA'
                confidence = 0.0
                strength = 0.0
            
            result = {
                'trend': trend,
                'confidence': confidence,
                'strength': strength,
                'timestamp': datetime.now().isoformat()
            }
            
            self.trend_signals[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi trong trend analysis cho {symbol}: {e}")
            return {'trend': 'ERROR', 'confidence': 0.0, 'strength': 0.0}
    
    def get_trend_recommendation(self, symbol: str) -> dict:
        """Lấy recommendation từ trend analysis"""
        if symbol in self.trend_signals:
            signal = self.trend_signals[symbol]
            return {
                'action': 'HOLD' if signal['trend'] == 'SIDEWAYS' else ('BUY' if signal['trend'] == 'UPTREND' else 'SELL'),
                'confidence': signal['confidence'],
                'reasoning': f"Trend analysis shows {signal['trend']} with {signal['confidence']:.2f} confidence"
            }
        return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'No trend data available'}

class RiskSpecialistAgent:
    """Specialist Agent cho risk assessment và position sizing"""
    
    def __init__(self):
        self.logger = LOG_MANAGER.get_logger('RiskSpecialistAgent')
        self.risk_assessments = {}
        
    async def assess_risk(self, symbol: str, data: dict, portfolio_value: float) -> dict:
        """Đánh giá risk cho symbol"""
        try:
            if symbol not in data or data[symbol].empty:
                return {'risk_level': 'HIGH', 'position_size': 0.0, 'confidence': 0.0}
            
            df = data[symbol].copy()
            
            # Risk metrics
            if len(df) >= 20:
                # Volatility (ATR)
                df['high_low'] = df['high'] - df['low']
                df['high_close'] = abs(df['high'] - df['close'].shift(1))
                df['low_close'] = abs(df['low'] - df['close'].shift(1))
                df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
                atr = df['tr'].rolling(window=14).mean().iloc[-1]
                current_price = df['close'].iloc[-1]
                
                # Volatility percentage
                volatility_pct = (atr / current_price) * 100
                
                # Risk level determination
                if volatility_pct < 1.0:
                    risk_level = 'LOW'
                    position_size = 0.02  # 2%
                elif volatility_pct < 2.0:
                    risk_level = 'MEDIUM'
                    position_size = 0.015  # 1.5%
                else:
                    risk_level = 'HIGH'
                    position_size = 0.01   # 1%
                
                # Recent drawdown
                recent_high = df['close'].rolling(window=20).max().iloc[-1]
                drawdown = ((recent_high - current_price) / recent_high) * 100
                
                confidence = 0.8 if len(df) >= 50 else 0.6
                
            else:
                risk_level = 'HIGH'
                position_size = 0.005  # 0.5% default
                volatility_pct = 5.0
                drawdown = 0.0
                confidence = 0.0
            
            result = {
                'risk_level': risk_level,
                'position_size': position_size,
                'volatility_pct': volatility_pct,
                'drawdown': drawdown,
                'confidence': confidence,
                'recommended_stop_loss': position_size * 0.5,  # 50% of position as stop loss
                'timestamp': datetime.now().isoformat()
            }
            
            self.risk_assessments[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi trong risk assessment cho {symbol}: {e}")
            return {'risk_level': 'HIGH', 'position_size': 0.0, 'confidence': 0.0}
    
    def get_risk_recommendation(self, symbol: str) -> dict:
        """Lấy risk recommendation"""
        if symbol in self.risk_assessments:
            assessment = self.risk_assessments[symbol]
            return {
                'max_position_size': assessment['position_size'],
                'risk_level': assessment['risk_level'],
                'stop_loss_size': assessment['recommended_stop_loss'],
                'confidence': assessment['confidence']
            }
        return {
            'max_position_size': 0.01,  # Default 1%
            'risk_level': 'MEDIUM',
            'stop_loss_size': 0.005,
            'confidence': 0.0
        }

class NewsSpecialistAgent:
    """Specialist Agent cho news và sentiment analysis"""
    
    def __init__(self, news_manager):
        self.logger = LOG_MANAGER.get_logger('NewsSpecialistAgent')
        self.news_manager = news_manager
        self.sentiment_cache = {}
        
    async def analyze_sentiment(self, symbol: str, hours_back: int = 24) -> dict:
        """Phân tích sentiment từ news"""
        try:
            if not self.news_manager:
                return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'impact': 'LOW'}
            
            # Get latest news
            news_data = await self.news_manager.fetch_news_sentiment(symbol, hours_back)
            
            if not news_data or not news_data.get('latest_news_items'):
                return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'impact': 'LOW'}
            
            # Analyze sentiment using Gemini
            sentiment_scores = []
            impact_keywords = ['fomc', 'nfp', 'gdp', 'inflation', 'interest rate', 'fed', 'central bank']
            
            for article in news_data['latest_news_items'][:10]:  # Limit to 10 articles
                text = article.get('text', '')
                title = text  # Use text as both title and description
                description = text
                
                # Check for high impact keywords
                high_impact = any(keyword in (title + description).lower() for keyword in impact_keywords)
                
                # Simple sentiment analysis (positive/negative keywords)
                text = f"{title} {description}".lower()
                positive_words = ['positive', 'growth', 'strong', 'increase', 'rise', 'bullish', 'surge', 'gains']
                negative_words = ['negative', 'decline', 'weak', 'decrease', 'fall', 'bearish', 'drop', 'loss']
                
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                if positive_count > negative_count:
                    sentiment_scores.append(1.0 if high_impact else 0.5)
                elif negative_count > positive_count:
                    sentiment_scores.append(-1.0 if high_impact else -0.5)
                else:
                    sentiment_scores.append(0.0)
            
            if not sentiment_scores:
                return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'impact': 'LOW'}
            
            # Calculate overall sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_sentiment > 0.3:
                sentiment = 'POSITIVE'
                confidence = min(1.0, abs(avg_sentiment))
            elif avg_sentiment < -0.3:
                sentiment = 'NEGATIVE'
                confidence = min(1.0, abs(avg_sentiment))
            else:
                sentiment = 'NEUTRAL'
                confidence = 0.5
            
            impact = 'HIGH' if any(abs(score) >= 0.8 for score in sentiment_scores) else 'MEDIUM' if confidence > 0.6 else 'LOW'
            
            result = {
                'sentiment': sentiment,
                'confidence': confidence,
                'impact': impact,
                'news_count': len(news_data['latest_news_items']),
                'timestamp': datetime.now().isoformat()
            }
            
            self.sentiment_cache[symbol] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi trong sentiment analysis cho {symbol}: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0, 'impact': 'LOW'}
    
    def get_sentiment_recommendation(self, symbol: str) -> dict:
        """Lấy sentiment recommendation"""
        if symbol in self.sentiment_cache:
            sentiment_data = self.sentiment_cache[symbol]
            return {
                'sentiment': sentiment_data['sentiment'],
                'confidence': sentiment_data['confidence'],
                'impact': sentiment_data['impact'],
                'news_count': sentiment_data['news_count']
            }
        return {
            'sentiment': 'NEUTRAL',
            'confidence': 0.0,
            'impact': 'LOW',
            'news_count': 0
        }

class MasterAgent:
    """Master Agent điều phối tất cả specialist agents và AI models"""
    
    def __init__(self, trend_agent, news_agent, risk_agent, ensemble_model=None, lstm_model=None, rl_agent=None):
        self.logger = LOG_MANAGER.get_logger('MasterAgent')
        self.trend_agent = trend_agent
        self.news_agent = news_agent
        self.risk_agent = risk_agent
        self.ensemble_model = ensemble_model
        self.lstm_model = lstm_model
        self.rl_agent = rl_agent
        self.decisions = {}
        self.ensemble_weight = 0.4
        self.lstm_weight = 0.3
        self.expert_weight = 0.3
        self.cycle_count = 0
        
    async def make_decision(self, symbol: str, data: dict, portfolio_value: float) -> dict:
        """Đưa ra quyết định trading cuối cùng sử dụng Ensemble AI và Expert Systems"""
        try:
            self.logger.info(f"🤖 Master Agent đang phân tích {symbol} với AI models...")
            
            # 1. Collect insights từ expert specialist agents
            trend_insight = await self.trend_agent.analyze_trend(symbol, data)
            sentiment_insight = await self.news_agent.analyze_sentiment(symbol)
            risk_assessment = await self.risk_agent.assess_risk(symbol, data, portfolio_value)
            
            # 2. Get AI Model predictions
            ensemble_prediction = None
            lstm_prediction = None
            rl_action = None
            
            # Ensemble Model prediction
            if self.ensemble_model and hasattr(self.ensemble_model, 'predict'):
                try:
                    # Check if ensemble model is properly trained
                    if not self.ensemble_model.models or not self.ensemble_model.meta_model:
                        self.logger.warning("Ensemble model chưa được huấn luyện")
                        ensemble_prediction = None
                    else:
                        # Convert data to DataFrame for ensemble prediction
                        import pandas as pd
                        features_df = pd.DataFrame([data])
                        ensemble_pred, ensemble_conf = self.ensemble_model.predict(features_df)
                        ensemble_prediction = {
                            'action': 'BUY' if ensemble_pred[0] > 0.5 else 'SELL' if ensemble_pred[0] < 0.3 else 'HOLD',
                            'confidence': ensemble_conf[0],
                            'probability': ensemble_pred[0]
                        }
                        self.logger.info(f"🎯 Ensemble prediction: {ensemble_prediction}")
                except Exception as e:
                    self.logger.warning(f"Ensemble model error: {e}")
                    ensemble_prediction = None
            
            # LSTM Model prediction
            if self.lstm_model and hasattr(self.lstm_model, 'predict'):
                try:
                    # Check if LSTM model is trained
                    if not self.lstm_model.is_trained:
                        self.logger.warning("LSTM model chưa được huấn luyện")
                        lstm_prediction = None
                    else:
                        lstm_pred, lstm_conf = self.lstm_model.predict(data)
                        
                        # Handle array inputs properly
                        if isinstance(lstm_pred, np.ndarray):
                            lstm_pred_val = lstm_pred[0] if len(lstm_pred) > 0 else 0.5
                        else:
                            lstm_pred_val = float(lstm_pred)
                            
                        if isinstance(lstm_conf, np.ndarray):
                            lstm_conf_val = lstm_conf[0] if len(lstm_conf) > 0 else 0.5
                        else:
                            lstm_conf_val = float(lstm_conf)
                        
                        lstm_prediction = {
                            'action': 'BUY' if lstm_pred_val > 0.6 else 'SELL' if lstm_pred_val < 0.4 else 'HOLD',
                            'confidence': lstm_conf_val,
                            'probability': lstm_pred_val
                        }
                        self.logger.info(f"🧠 LSTM prediction: {lstm_prediction}")
                except Exception as e:
                    self.logger.warning(f"LSTM model error: {e}")
                    lstm_prediction = None
            
            # RL Agent action
            if self.rl_agent and self.rl_agent.model:
                try:
                    env_state = [portfolio_value, data.get('close', 0), risk_assessment['risk_level'] == 'HIGH']
                    action, _ = self.rl_agent.model.predict(env_state, deterministic=True)
                    rl_action = {
                        'action': 'BUY' if action == 0 else 'SELL' if action == 1 else 'HOLD',
                        'confidence': 0.7,  # Default RL confidence
                        'action_id': action
                    }
                    self.logger.info(f"🤖 RL Agent action: {rl_action}")
                except Exception as e:
                    self.logger.warning(f"RL Agent error: {e}")
            
            # 3. Weighted Ensemble Decision Making
            expert_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            # Expert system votes
            trend_recommendation = self.trend_agent.get_trend_recommendation(symbol)
            expert_votes[trend_recommendation['action']] += trend_recommendation['confidence'] * self.expert_weight
            
            sentiment_recommendation = self.news_agent.get_sentiment_recommendation(symbol)
            if sentiment_recommendation['sentiment'] == 'POSITIVE':
                expert_votes['BUY'] += sentiment_recommendation['confidence'] * self.expert_weight
            elif sentiment_recommendation['sentiment'] == 'NEGATIVE':
                expert_votes['SELL'] += sentiment_recommendation['confidence'] * self.expert_weight
            else:
                expert_votes['HOLD'] += sentiment_recommendation['confidence'] * self.expert_weight
            
            # Combine AI predictions with expert systems
            if ensemble_prediction:
                expert_votes[ensemble_prediction['action']] += ensemble_prediction['confidence'] * self.ensemble_weight
            if lstm_prediction:
                expert_votes[lstm_prediction['action']] += lstm_prediction['confidence'] * self.lstm_weight
            if rl_action:
                expert_votes[rl_action['action']] += rl_action['confidence'] * 0.3  # RL weight
            
            # 4. Final weighted decision
            best_action = max(expert_votes, key=expert_votes.get)
            confidence = expert_votes[best_action] / max(sum(expert_votes.values()), 0.1)
            
            # Risk-based override
            max_position_size = risk_assessment['position_size']
            if risk_assessment['risk_level'] == 'HIGH' and best_action != 'HOLD':
                best_action = 'HOLD'
                confidence *= 0.5
                self.logger.info(f"⚠️ Risk override: HIGH risk forces HOLD")
            
            # Calculate ensemble confidence boost
            if ensemble_prediction and lstm_prediction:
                ai_consensus = abs(ensemble_prediction['probability'] - 0.5) + abs(lstm_prediction['probability'] - 0.5)
                confidence = min(confidence + ai_consensus * 0.2, 1.0)
            
            decision = {
                'action': best_action,
                'confidence': min(confidence, 1.0),
                'position_size': min(max_position_size, 0.02),
                'ai_predictions': {
                    'ensemble': ensemble_prediction,
                    'lstm': lstm_prediction,
                    'rl': rl_action
                },
                'reasoning': {
                    'trend': trend_recommendation,
                    'sentiment': sentiment_recommendation,
                    'risk': {
                        'level': risk_assessment['risk_level'],
                        'max_size': risk_assessment['position_size']
                    },
                    'weights': {
                        'ensemble': self.ensemble_weight,
                        'lstm': self.lstm_weight,
                        'expert': self.expert_weight
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.decisions[symbol] = decision
            self.logger.info(f"✅ Master Agent quyết định: {best_action} {symbol} với confidence {confidence:.2f} (AI-Enhanced)")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Lỗi trong decision making cho {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'position_size': 0.0,
                'reasoning': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }
    
    def get_last_decision(self, symbol: str) -> dict:
        """Lấy decision gần nhất cho symbol"""
        return self.decisions.get(symbol, {
            'action': 'HOLD',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        })
    
    async def analyze_and_decide(self, enriched_data: dict, portfolio_status: dict) -> dict:
        """Analyze enriched data from all symbols and make comprehensive trading decisions"""
        try:
            self.logger.info("🤖 Master Agent analyzing enriched data for trading decisions...")
            symbols_with_data = list(enriched_data.keys())
            self.logger.info(f"📊 Symbols có dữ liệu để phân tích: {', '.join(symbols_with_data)}")
            
            # Increment cycle count for training triggers
            self.cycle_count += 1
            
            decisions = {}
            buy_signals = []
            sell_signals = []
            
            # Analyze each symbol in enriched data
            for symbol, features_data in enriched_data.items():
                if features_data is None or features_data.empty:
                    continue
                    
                # Get latest row for decision making
                latest_data = features_data.iloc[-1].to_dict()
                
                # Trigger AI training if we have enough data (every 500 cycles hoặc khi performance thấp)
                if len(features_data) >= 50 and self.cycle_count % 500 == 0:
                    self.logger.info(f"🔄 Triggering AI training cho {symbol}...")
                    training_result = await self.trigger_ai_training(symbol, features_data)
                    self.logger.info(f"🎓 Training result: {training_result.get('status', 'unknown')}")
                
                # Make individual decision for this symbol
                individual_decision = await self.make_decision(symbol, latest_data, portfolio_status.get('total_value', 100000))
                decisions[symbol] = individual_decision
                
                # Collect signals
                if individual_decision['action'] == 'BUY' and individual_decision['confidence'] > 0.6:
                    buy_signals.append({
                        'symbol': symbol,
                        'confidence': individual_decision['confidence'],
                        'position_size': individual_decision.get('position_size', 0.02)
                    })
                elif individual_decision['action'] == 'SELL' and individual_decision['confidence'] > 0.6:
                    sell_signals.append({
                        'symbol': symbol,
                        'confidence': individual_decision['confidence'],
                        'position_size': individual_decision.get('position_size', 0.02)
                    })
            
            # Calculate overall signal strength
            total_confidence = sum(s['confidence'] for s in buy_signals + sell_signals)
            signal_count = len(buy_signals) + len(sell_signals)
            signal_strength = total_confidence / max(signal_count, 1) if signal_count > 0 else 0
            
            # Determine market regime
            market_regime = 'NEUTRAL'
            if len(buy_signals) > len(sell_signals) and signal_strength > 0.7:
                market_regime = 'BULLISH'
            elif len(sell_signals) > len(buy_signals) and signal_strength > 0.7:
                market_regime = 'BEARISH'
            
            # Comprehensive decision
            comprehensive_decision = {
                'signal_strength': signal_strength,
                'confidence': signal_strength,
                'market_regime': market_regime,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'individual_decisions': decisions,
                'portfolio_status': portfolio_status,
                'total_signals': len(buy_signals) + len(sell_signals),
                'recommended_action': 'BUY' if market_regime == 'BULLISH' else 'SELL' if market_regime == 'BEARISH' else 'HOLD',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Master Agent analysis complete: {market_regime} market, {signal_strength:.2f} signal strength, {len(buy_signals)} buy, {len(sell_signals)} sell")
            
            return comprehensive_decision
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi trong analyze_and_decide: {e}")
            return {
                'signal_strength': 0.0,
                'confidence': 0.0,
                'market_regime': 'NEUTRAL',
                'buy_signals': [],
                'sell_signals': [],
                'individual_decisions': {},
                'portfolio_status': portfolio_status,
                'total_signals': 0,
                'recommended_action': 'HOLD',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def trigger_ai_training(self, symbol: str, recent_data: pd.DataFrame) -> dict:
        """Trigger AI model training khi có đủ dữ liệu"""
        try:
            if recent_data.empty or len(recent_data) < 50:
                return {'status': 'insufficient_data', 'data_points': len(recent_data)}
            
            self.logger.info(f"🎓 Training AI models cho {symbol} với {len(recent_data)} data points...")
            
            # Prepare features và labels cho training
            features = recent_data.drop(['close'], axis=1, errors='ignore')
            labels = (recent_data['close'].shift(-1) > recent_data['close']).astype(int)[:-1]
            features = features.iloc[:-1]  # Remove last row để match labels
            
            training_results = {}
            
            # Train Ensemble Model
            if self.ensemble_model and hasattr(self.ensemble_model, 'train_ensemble'):
                try:
                    ensemble_scores = self.ensemble_model.train_ensemble(features, labels)
                    training_results['ensemble'] = ensemble_scores
                    self.logger.info(f"✅ Ensemble training completed - Accuracy: {ensemble_scores.get('accuracy', 0):.3f}")
                except Exception as e:
                    self.logger.error(f"Ensemble training failed: {e}")
                    training_results['ensemble'] = {'error': str(e)}
            
            # Train LSTM Model
            if self.lstm_model and hasattr(self.lstm_model, 'train'):
                try:
                    lstm_scores = self.lstm_model.train(features, labels)
                    training_results['lstm'] = lstm_scores
                    self.logger.info(f"✅ LSTM training completed - Validation Accuracy: {lstm_scores.get('val_accuracy', 0):.3f}")
                except Exception as e:
                    self.logger.error(f"LSTM training failed: {e}")
                    training_results['lstm'] = {'error': str(e)}
            
            # Train RL Agent
            if self.rl_agent and self.rl_agent.model:
                try:
                    self.rl_agent.train(total_timesteps=50000)
                    training_results['rl'] = {'status': 'trained', 'timesteps': 10000}
                    self.logger.info("✅ RL Agent training completed")
                except Exception as e:
                    self.logger.error(f"RL training failed: {e}")
                    training_results['rl'] = {'error': str(e)}
            
            # Cập nhật model weights nếu training thành công
            trained_models = [k for k, v in training_results.items() if 'error' not in str(v)]
            if len(trained_models) >= 2:
                # Boost confidence weight của các models trained
                self.ensemble_weight = 0.5
                self.lstm_weight = 0.4
                self.expert_weight = 0.1
                self.logger.info(f"🎯 Model weights updated after training: {trained_models}")
            
            return {
                'status': 'training_completed',
                'trained_models': trained_models,
                'results': training_results,
                'updated_weights': {
                    'ensemble': self.ensemble_weight,
                    'lstm': self.lstm_weight,
                    'expert': self.expert_weight
                }
            }
            
        except Exception as e:
            self.logger.error(f"AI training failed: {e}")
            return {'status': 'error', 'message': str(e)}

class RLAgent:
    """RL Agent sử dụng PPO từ stable-baselines3"""
    
    def __init__(self, environment: PortfolioEnvironment):
        self.environment = environment
        self.logger = LOG_MANAGER.get_logger('RLAgent')
        self.model = None
        self.training_history = []
    
    def create_model(self, policy: str = "MlpPolicy"):
        """Tạo PPO model"""
        
        # Callback cho training monitoring
        callback = TrainingCallback(check_freq=5000)
        
        # Create PPO model
        self.model = PPO(
            policy=policy,
            env=self.environment,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./rl_tensorboard/"
        )
        
        self.logger.info("Đã tạo PPO model")
        return self.model
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Huấn luyện RL agent"""
        
        if not self.model:
            self.create_model()
        
        self.logger.info(f"Bắt đầu training RL agent trong {total_timesteps} steps")
        
        # Train model
        with self.model.env.envs[0] as env:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=TrainingCallback(),
                progress_bar=True
            )
        
        # Save model
        self.model.save("ppo_trading_agent")
        self.logger.info("Đã hoàn thành RL training")
        
        return {
            'total_timesteps': total_timesteps,
            'training_completed': True
        }
    
    def predict(self, observation) -> Tuple[np.ndarray, Dict]:
        """Dự đoán action với RL model"""
        
        if not self.model:
            self.logger.warning("RL model chưa được train")
            return np.zeros(self.environment.n_symbols), {}
        
        action, _states = self.model.predict(observation, deterministic=True)
        return action, {'action_distribution': action.tolist()}
    
    def evaluate_performance(self, n_episodes: int = 10) -> Dict[str, float]:
        """Đánh giá performance của trained agent"""
        
        if not self.model:
            return {'error': 'Model chưa được train'}
        
        episode_returns = []
        episode_sharpe_ratios = []
        
        for episode in range(n_episodes):
            obs = self.environment.reset()
            episode_return = 0
            returns = []
            
            done = False
            while not done:
                action, _ = self.predict(obs)
                obs, reward, done, info = self.environment.step(action)
                episode_return += reward
                returns.append(info['daily_return'])
            
            episode_returns.append(episode_return)
            
            # Calculate Sharpe ratio
            if returns:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                episode_sharpe_ratios.append(sharpe)
        
        performance_metrics = {
            'mean_return': np.mean(episode_returns),
            'return_std': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpe_ratios),
            'episodes_evaluated': n_episodes
        }
        
        self.logger.info(f"RL Performance - Mean Return: {performance_metrics['mean_return']:.2f}, "
                        f"Sharpe: {performance_metrics['mean_sharpe']:.3f}")
        
        return performance_metrics

# ===== AUTO RETRAIN MANAGER VÀ CONCEPT DRIFT DETECTION =====
class ConceptDriftDetector:
    """Phát hiện concept drift trong dữ liệu thị trường"""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.logger = LOG_MANAGER.get_logger('AutoRetrainManager')
        self.reference_distribution = None
        self.drift_detected = False
    
    def update_reference(self, data_sample: np.ndarray):
        """Cập nhật distribution tham chiếu"""
        if self.reference_distribution is None:
            self.reference_distribution = np.mean(data_sample, axis=0)
        else:
            # Exponential moving average để cập nhật reference
            alpha = 0.1
            new_mean = np.mean(data_sample, axis=0)
            self.reference_distribution = alpha * new_mean + (1 - alpha) * self.reference_distribution
    
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, bool]:
        """Phát hiện drift trong dữ liệu mới"""
        
        results = {
            'distribution_drift': False,
            'statistical_drift': False,
            'overall_drift': False
        }
        
        try:
            if self.reference_distribution is None:
                self.update_reference(new_data)
                return results
            
            # 1. Distribution drift detection
            dist_drift = False
            if len(new_data) >= self.window_size:
                current_mean = np.mean(new_data[-self.window_size:], axis=0)
                
                # Calculate distance between distributions
                distribution_distance = np.linalg.norm(current_mean - self.reference_distribution)
                dist_drift = distribution_distance > self.threshold
            
            results['distribution_drift'] = dist_drift
            
            # 2. Statistical drift detection
            statistical_drift = False
            if len(new_data) >= self.window_size * 2:
                recent_data = new_data[-self.window_size:]
                historical_data = new_data[-self.window_size*2:-self.window_size]
                
                # T-test để so sánh means
                from scipy import stats
                try:
                    recent_mean = np.mean(recent_data)
                    historical_mean = np.mean(historical_data)
                    
                    _, p_value = stats.ttest_ind(recent_data.flatten(), historical_data.flatten())
                    statistical_drift = p_value < 0.05 and abs(recent_mean - historical_mean) > self.threshold
                except:
                    statistical_drift = False
            
            results['statistical_drift'] = statistical_drift
            
            # 3. Overall drift assessment
            results['overall_drift'] = dist_drift or statistical_drift
            
            if results['overall_drift']:
                self.logger.warning("Concept drift detected!")
                self.drift_detected = True
            
        except Exception as e:
            self.logger.error(f"Lỗi trong drift detection: {e}")
        
        return results

class OnlineLearningManager:
    """Quản lý online learning với river framework"""
    
    def __init__(self):
        self.logger = LOG_MANAGER.get_logger('AutoRetrainManager')
        self.online_models = {}
        
        # Initialize adaptive models cho từng symbol
        self.online_models = {}
        for symbol in Config.SYMBOLS:
            self.online_models[symbol] = {
                'linear_model': linear_model.LogisticRegression(),
                'preprocessor': preprocessing.StandardScaler(),
                'anomaly_detector': anomaly.OneClassSVM(),
                'performance_history': [],
                'initialized': False
            }
    
    def _numpy_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to dict format for River framework"""
        if len(features.shape) == 1:
            # 1D array -> dict with numeric keys
            return {f"feature_{i}": float(features[i]) for i in range(len(features))}
        elif len(features.shape) == 2 and features.shape[0] == 1:
            # 2D array with 1 row -> flatten to 1D
            flattened = features.flatten()
            return {f"feature_{i}": float(flattened[i]) for i in range(len(flattened))}
        else:
            # Complex array -> flatten and truncate
            flattened = features.flatten()[:50]  # Limit to 50 features
            return {f"feature_{i}": float(flattened[i]) for i in range(len(flattened))}
    
    def update_model(self, symbol: str, features: np.ndarray, target: float):
        """Cập nhật online model với data mới"""
        
        try:
            if symbol not in self.online_models:
                return
            
            model_info = self.online_models[symbol]
            
            # Convert numpy array to dict for River framework
            features_dict = self._numpy_to_dict(features)
            
            # Preprocess features
            if model_info['preprocessor'] is None:
                self.logger.warning(f"Preprocessor is None for {symbol}, skipping model update")
                return
            
            # Handle case where learn_one returns None (common with River framework)
            preprocessor_result = model_info['preprocessor'].learn_one(features_dict)
            if preprocessor_result is None:
                # For StandardScaler, we can still transform without learning first
                try:
                    scaled_features = model_info['preprocessor'].transform_one(features_dict)
                    model_info['initialized'] = True  # Mark as initialized on successful transform
                except Exception as e:
                    self.logger.warning(f"Could not transform features for {symbol}: {e}. Model not yet initialized with enough data.")
                    return
            else:
                scaled_features = preprocessor_result.transform_one(features_dict)
            
            # Update anomaly detector
            model_info['anomaly_detector'].learn_one(scaled_features)
            
            # Update linear model
            model_info['linear_model'].learn_one(scaled_features, target)
            
            # Track performance
            prediction = model_info['linear_model'].predict_one(scaled_features)
            error = abs(prediction - target)
            model_info['performance_history'].append(error)
            
            # Keep only last 100 performance records
            if len(model_info['performance_history']) > 100:
                model_info['performance_history'] = model_info['performance_history'][-100:]
            
            if len(model_info['performance_history']) % 10 == 0:
                avg_error = np.mean(model_info['performance_history'][-10:])
                self.logger.info(f"Online model {symbol} - Average error: {avg_error:.3f}")
            
        except Exception as e:
            self.logger.error(f"Lỗi cập nhật online model cho {symbol}: {e}")
    
    def get_prediction(self, symbol: str, features: np.ndarray) -> Tuple[float, float]:
        """Lấy prediction từ online model"""
        
        try:
            if symbol not in self.online_models:
                return 0.0, 0.0
            
            model_info = self.online_models[symbol]
            
            # Convert numpy array to dict for River framework
            features_dict = self._numpy_to_dict(features)
            if model_info['preprocessor'] is None:
                self.logger.warning(f"Preprocessor is None for {symbol}, returning default prediction")
                return 0.0
            
            try:
                scaled_features = model_info['preprocessor'].transform_one(features_dict)
            except Exception as e:
                self.logger.warning(f"Could not transform features for {symbol}: {e}. Model not yet initialized.")
                return 0.0
            
            prediction = model_info['linear_model'].predict_one(scaled_features)
            confidence = 1.0 - np.mean(model_info['performance_history'][-10:]) if model_info['performance_history'] else 0.5
            
            return prediction, min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Lỗi prediction từ online model {symbol}: {e}")
            return 0.0, 0.0
    
    def detect_anomaly(self, symbol: str, features: np.ndarray) -> bool:
        """Phát hiện anomaly trong data"""
        
        try:
            if symbol not in self.online_models:
                return False
            
            model_info = self.online_models[symbol]
            if model_info['preprocessor'] is None:
                self.logger.warning(f"Preprocessor is None for {symbol}, returning no anomaly")
                return False
            
            try:
                scaled_features = model_info['preprocessor'].transform_one(features)
            except Exception as e:
                self.logger.warning(f"Could not transform features for {symbol}: {e}. Model not yet initialized.")
                return False
            
            # Check if current data is anomaly
            anomaly_score = model_info['anomaly_detector'].score_one(scaled_features)
            
            # Threshold for anomaly detection (adjust based on domain knowledge)
            return anomaly_score < -1.0  # Negative score indicates anomaly
            
        except Exception as e:
            self.logger.error(f"Lỗi anomaly detection cho {symbol}: {e}")
            return False

class AutoRetrainManager:
    """Quản lý tự động retrain models khi có drift hoặc performance giảm"""
    
    def __init__(self, ensemble_model: EnsembleModel, lstm_model: LSTMModel):
        self.ensemble_model = ensemble_model
        self.lstm_model = lstm_model
        self.drift_detector = ConceptDriftDetector()
        self.online_manager = OnlineLearningManager()
        self.logger = LOG_MANAGER.get_logger('AutoRetrainManager')
        
        # Performance tracking
        self.performance_history = {
            'ensemble': [],
            'lstm': [],
            'overall': []
        }
        self.last_retrain_time = datetime.now()
        self.retrain_threshold = 0.05  # Retrain nếu performance giảm >5%
        
        # Model versions và backup
        self.model_versions = 0
        self.model_backups = {}
    
    def evaluate_model_performance(self, models_data: Dict[str, Dict]) -> Dict[str, float]:
        """Đánh giá performance hiện tại của models"""
        
        performance_metrics = {
            'ensemble_accuracy': 0.0,
            'lstm_accuracy': 0.0,
            'overall_performance': 0.0,
            'performance_trend': 'stable'
        }
        
        try:
            # Simulate model evaluation (trong thực tế sẽ có real data)
            performance_metrics['ensemble_accuracy'] = np.random.uniform(0.6, 0.8)
            performance_metrics['lstm_accuracy'] = np.random.uniform(0.6, 0.9)
            
            # Overall performance là weighted average
            performance_metrics['overall_performance'] = (
                0.6 * performance_metrics['ensemble_accuracy'] + 
                0.4 * performance_metrics['lstm_accuracy']
            )
            
            # Track performance history
            self.performance_history['ensemble'].append(performance_metrics['ensemble_accuracy'])
            self.performance_history['lstm'].append(performance_metrics['lstm_accuracy'])
            self.performance_history['overall'].append(performance_metrics['overall_performance'])
            
            # Keep only last 50 records
            for key in self.performance_history:
                if len(self.performance_history[key]) > 50:
                    self.performance_history[key] = self.performance_history[key][-50:]
            
            # Calculate performance trend
            if len(self.performance_history['overall']) >= 10:
                recent_avg = np.mean(self.performance_history['overall'][-5:])
                older_avg = np.mean(self.performance_history['overall'][-10:-5])
                
                if recent_avg > older_avg + 0.02:
                    performance_metrics['performance_trend'] = 'improving'
                elif recent_avg < older_avg - 0.02:
                    performance_metrics['performance_trend'] = 'declining'
                else:
                    performance_metrics['performance_trend'] = 'stable'
            
            self.logger.info(f"Model Performance - Overall: {performance_metrics['overall_performance']:.3f}, "
                            f"Trend: {performance_metrics['performance_trend']}")
            
        except Exception as e:
            self.logger.error(f"Lỗi đánh giá performance: {e}")
        
        return performance_metrics
    
    def check_retrain_conditions(self, new_data: pd.DataFrame) -> Dict[str, bool]:
        """Kiểm tra điều kiện cần retrain models"""
        
        retrain_conditions = {
            'performance_decline': False,
            'concept_drift': False,
            'time_based': False,
            'should_retrain': False
        }
        
        try:
            # 1. Check performance decline
            if len(self.performance_history['overall']) >= 10:
                recent_performance = np.mean(self.performance_history['overall'][-5:])
                historical_performance = np.mean(self.performance_history['overall'][-10:-5])
                
                performance_drop = historical_performance - recent_performance
                retrain_conditions['performance_decline'] = performance_drop > self.retrain_threshold
            
            # 2. Check concept drift
            np_data = new_data.select_dtypes(include=[np.number]).values
            drift_results = self.drift_detector.detect_drift(np_data)
            retrain_conditions['concept_drift'] = drift_results['overall_drift']
            
            # 3. Check time-based retraining (mỗi 24h)
            hours_since_last_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            retrain_conditions['time_based'] = hours_since_last_retrain > 24
            
            # 4. Overall decision
            retrain_conditions['should_retrain'] = (
                retrain_conditions['performance_decline'] or 
                retrain_conditions['concept_drift'] or 
                retrain_conditions['time_based']
            )
            
            if retrain_conditions['should_retrain']:
                self.logger.warning(f"Retrain conditions met - "
                                  f"Performance: {retrain_conditions['performance_decline']}, "
                                  f"Drift: {retrain_conditions['concept_drift']}, "
                                  f"Time: {retrain_conditions['time_based']}")
            
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra retrain conditions: {e}")
        
        return retrain_conditions
    
    def execute_retrain(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       symbol: str = "ALL") -> Dict[str, Any]:
        """Thực hiện retrain models"""
        
        retrain_results = {
            'retrain_success': False,
            'new_accuracy': 0.0,
            'improvement': 0.0,
            'models_updated': [],
            'retain_time': datetime.now().isoformat()
        }
        
        try:
            self.logger.info(f"Bắt đầu retrain cho {symbol}")
            
            # Backup current models
            self.model_versions += 1
            backup_key = f"version_{self.model_versions}_{symbol}"
            
            if hasattr(self.ensemble_model, 'models'):
                self.model_backups[backup_key] = self.ensemble_model.models.copy()
            
            # Store old performance for comparison
            old_performance = np.mean(self.performance_history['overall'][-5:]) if self.performance_history['overall'] else 0.0
            
            # Retrain ensemble model
            try:
                ensemble_scores = self.ensemble_model.train_ensemble(X_train, y_train)
                retrain_results['models_updated'].append('ensemble')
                self.logger.info(f"Ensemble retrain completed - F1: {ensemble_scores.get('f1', 0):.3f}")
            except Exception as e:
                self.logger.error(f"Lỗi retrain ensemble: {e}")
            
            # Retrain LSTM model
            try:
                lstm_scores = self.lstm_model.train(X_train, y_train)
                retrain_results['models_updated'].append('lstm')
                self.logger.info(f"LSTM retrain completed - Val Acc: {lstm_scores.get('val_accuracy', 0):.3f}")
            except Exception as e:
                self.logger.error(f"Lỗi retrain LSTM: {e}")
            
            # Update performance tracking
            if retrain_results['models_updated']:
                new_performance = np.max([
                    ensemble_scores.get('f1', 0), 
                    lstm_scores.get('val_accuracy', 0)
                ])
                
                retrain_results['new_accuracy'] = new_performance
                retrain_results['improvement'] = new_performance - old_performance
                
                # Update history
                self.performance_history['ensemble'].append(ensemble_scores.get('f1', 0))
                self.performance_history['lstm'].append(lstm_scores.get('val_accuracy', 0))
                self.performance_history['overall'].append(new_performance)
                
                self.last_retrain_time = datetime.now()
                retrain_results['retrain_success'] = True
                
                self.logger.info(f"Retrain completed - Improvement: {retrain_results['improvement']:.3f}")
                
                # Nếu improved performance đáng kể, remove old backup
                if retrain_results['improvement'] > 0.02:
                    old_backups_to_remove = [k for k in self.model_backups.keys() if backup_key in k and k != backup_key]
                    for old_backup in old_backups_to_remove[-3:]:  # Keep only 3 most recent backups
                        del self.model_backups[old_backup]
            
        except Exception as e:
            self.logger.error(f"Lỗi trong retrain process: {e}")
        
        return retrain_results
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Lấy thông tin diagnostics về models"""
        
        diagnostics = {
            'performance_history': {
                'ensemble_mean': np.mean(self.performance_history['ensemble']) if self.performance_history['ensemble'] else 0,
                'lstm_mean': np.mean(self.performance_history['lstm']) if self.performance_history['lstm'] else 0,
                'overall_mean': np.mean(self.performance_history['overall']) if self.performance_history['overall'] else 0,
                'overall_trend': 'not_enough_data'
            },
            'drift_status': {
                'drift_detected': self.drift_detector.drift_detected,
                'total_drifts': 0
            },
            'retrain_info': {
                'version_count': self.model_versions,
                'last_retrain': self.last_retrain_time.isoformat(),
                'hours_since_retrain': (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            },
            'online_models': {}
        }
        
        # Performance trend analysis
        if len(self.performance_history['overall']) >= 5:
            recent_points = self.performance_history['overall'][-5:]
            slope = np.polyfit(range(len(recent_points)), recent_points, 1)[-1]
            
            if slope > 0.005:
                diagnostics['performance_history']['overall_trend'] = 'improving'
            elif slope < -0.005:
                diagnostics['performance_history']['overall_trend'] = 'declining'
            else:
                diagnostics['performance_history']['overall_trend'] = 'stable'
        
        # Online model diagnostics
        for symbol in Config.SYMBOLS:
            if symbol in self.online_manager.online_models:
                model_info = self.online_manager.online_models[symbol]
                diagnostics['online_models'][symbol] = {
                    'performance_record_count': len(model_info['performance_history']),
                    'recent_average_error': np.mean(model_info['performance_history'][-10:]) if model_info['performance_history'] else 0
                }
        
        return diagnostics# ===== OBSERVABILITY VÀ DISCORD NOTIFICATIONS =====
class DiscordNotificationManager:
    """Quản lý thông báo Discord với rich embeds"""
    
    def __init__(self):
        self.webhook_url = Config.DISCORD_WEBHOOK
        self.logger = LOG_MANAGER.get_logger('Discord')
        self.notification_queue = queue.Queue()
        self.rate_limit_delay = 1  # seconds between notifications
    
    async def send_signal_notification(self, symbol: str, signal_data: Dict[str, Any]):
        """Gửi thông báo signal mới"""
        
        embed_data = {
            "title": f"🚨 New Trading Signal - {symbol}",
            "color": 0x00ff00 if signal_data.get('action_type') == 'buy' else 0xff0000,
            "fields": [
                {"name": "Action", "value": signal_data['action_type'].upper(), "inline": True},
                {"name": "Signal Strength", "value": f"{signal_data.get('signal_strength', 0):.2f}", "inline": True},
                {"name": "Confidence", "value": f"{signal_data.get('confidence', 0):.2f}", "inline": True},
                {"name": "Market Regime", "value": signal_data.get('market_regime', 'unknown'), "inline": True},
                {"name": "Reason", "value": signal_data.get('decision_reason', 'No reason provided'), "inline": False}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Trading Bot AI/ML"}
        }
        
        await self._send_embed(embed_data)
    
    async def send_position_opened(self, symbol: str, position_data: Dict[str, Any]):
        """Gửi thông báo mở position"""
        
        embed_data = {
            "title": f"🟢 Position Opened - {symbol}",
            "color": 0x00ff00,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Size", "value": f"{position_data.get('size', 0):.2f}", "inline": True},
                {"name": "Entry Price", "value": f"{position_data.get('entry_price', 0):.4f}", "inline": True},
                {"name": "Stop Loss", "value": f"{position_data.get('stop_loss', 0):.4f}", "inline": True},
                {"name": "Take Profit", "value": f"{position_data.get('take_profit', 0):.4f}", "inline": True},
                {"name": "Risk", "value": f"{position_data.get('risk_percentage', 0):.1f}%", "inline": True}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Position Management"}
        }
        
        await self._send_embed(embed_data)
    
    async def send_position_closed(self, symbol: str, profit_loss: float, return_percentage: float):
        """Gửi thông báo đóng position"""
        
        emoji = "📈" if profit_loss >= 0 else "📉"
        color = 0x00ff00 if profit_loss >= 0 else 0xff0000
        
        embed_data = {
            "title": f"{emoji} Position Closed - {symbol}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "PnL", "value": f"${profit_loss:.2f}", "inline": True},
                {"name": "Return %", "value": f"{return_percentage:.2f}%", "inline": True},
                {"name": "Result", "value": "PROFIT" if profit_loss >= 0 else "LOSS", "inline": True}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Trade Result"}
        }
        
        await self._send_embed(embed_data)
    
    async def send_error_notification(self, error_type: str, error_message: str, symbol: str = ""):
        """Gửi thông báo lỗi hệ thống"""
        
        title_suffix = f" - {symbol}" if symbol else ""
        
        embed_data = {
            "title": f"⚠️ Error Alert{title_suffix}",
            "color": 0xff8800,
            "fields": [
                {"name": "Error Type", "value": error_type, "inline": True},
                {"name": "Message", "value": error_message[:1000], "inline": False},
                {"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "System Alert"}
        }
        
        await self._send_embed(embed_data)
    
    async def send_performance_report(self, metrics: Dict[str, float], model_performance: Dict[str, Any]):
        """Gửi báo cáo hiệu suất"""
        
        embed_data = {
            "title": "📊 Daily Performance Report",
            "color": 0x0099ff,
            "fields": [
                {"name": "Portfolio Value", "value": f"${metrics.get('total_value', 0):,.2f}", "inline": True},
                {"name": "Daily P&L", "value": f"${metrics.get('daily_pnl', 0):,.2f}", "inline": True},
                {"name": "Daily Return", "value": f"{metrics.get('daily_return', 0):.2f}%", "inline": True},
                {"name": "Sharpe Ratio", "value": f"{metrics.get('sharpe_ratio', 0):.3f}", "inline": True},
                {"name": "Max Drawdown", "value": f"{metrics.get('max_drawdown', 0):.2f}%", "inline": True},
                {"name": "Model Performance", "value": f"{model_performance.get('overall_performance', 0):.3f}", "inline": True}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Trading Bot Performance"}
        }
        
        await self._send_embed(embed_data)
    
    async def _send_embed(self, embed_data: Dict[str, Any]):
        """Gửi embed message tới Discord webhook"""
        
        try:
            payload = {"embeds": [embed_data]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 204:  # Discord webhook success
                        self.logger.info("Discord notification sent successfully")
                    else:
                        self.logger.warning(f"Discord notification failed: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Lỗi gửi Discord notification: {e}")
        
        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)

class AdvancedRiskManager:
    """Advanced Risk Management System"""
    
    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.logger = LOG_MANAGER.get_logger('RiskManager')
        
        # Portfolio settings
        self.max_total_exposure = 1.0  # 100% of portfolio
        self.max_position_size = 0.1   # 10% per position
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_ratio = 2.0   # 2:1 risk/reward
        
        # Position tracking
        self.open_positions = {}
        self.portfolio_value = 10000   # Starting portfolio value
        self.cash_balance = 10000      # Available cash
        
    def calculate_position_size(self, symbol: str, signal_strength: float, confidence: float, entry_price: float) -> Dict[str, Any]:
        """Calculate optimal position size based on risk parameters"""
        
        # Base position size calculation
        base_size = self.cash_balance * self.max_position_size
        
        # Adjust based on signal strength and confidence
        adjustment_factor = min(signal_strength * confidence, 1.0)
        position_size = base_size * adjustment_factor
        
        # Calculate stop loss and take profit
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price + (entry_price - stop_loss) * self.take_profit_ratio
        
        # Calculate risk percentage
        risk_amount = abs(position_size * (entry_price - stop_loss) / entry_price)
        risk_percentage = (risk_amount / self.portfolio_value) * 100
        
        return {
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_percentage': risk_percentage,
            'risk_amount': risk_amount
        }
    
    def validate_trade(self, symbol: str, position_data: Dict[str, Any]) -> bool:
        """Validate if trade meets risk criteria"""
        
        # Check if we already have a position in this symbol
        if symbol in self.open_positions:
            self.logger.warning(f"Already have position in {symbol}")
            return False
            
        # Check portfolio exposure
        current_exposure = self._calculate_portfolio_exposure()
        new_exposure = position_data.get('risk_amount', 0)
        
        if current_exposure + new_exposure > self.max_total_exposure * self.portfolio_value:
            self.logger.warning(f"Trade rejected: Portfolio exposure too high ({current_exposure + new_exposure:.2f})")
            return False
            
        # Check available cash
        if position_data.get('size', 0) > self.cash_balance:
            self.logger.warning(f"Trade rejected: Insufficient cash ({self.cash_balance:.2f})")
            return False
            
        return True
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position tracking"""
        
        position_data.update({
            'status': 'open',
            'timestamp': datetime.now(),
            'symbol': symbol
        })
        
        self.open_positions[symbol] = position_data
        
        # Update cash balance
        self.cash_balance -= position_data.get('size', 0)
        
        # Insert into database
        try:
            import sqlite3
            
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            # Determine position type from the data
            position_type = 'buy' if position_data.get('position_type', 'buy') == 'buy' else 'sell'
            
            # Insert new position record
            cursor.execute("""
                INSERT INTO positions (
                    symbol, status, position_type, size, entry_price, 
                    stop_loss, take_profit, entry_time, risk_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                'open',
                position_type,
                position_data.get('size', 0),
                position_data.get('entry_price', 0),
                position_data.get('stop_loss'),
                position_data.get('take_profit'),
                datetime.now().isoformat(),
                position_data.get('risk_percentage', 0)
            ))
            
            # Get the ID of the newly created record
            position_id = cursor.lastrowid
            
            # Store the position ID in the position data for future reference
            position_data['db_id'] = position_id
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Position entered into database with ID {position_id} for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to insert position into database for {symbol}: {e}")
        
        self.logger.info(f"Position updated for {symbol}: {position_data}")
    
    def close_position(self, symbol: str, exit_price: float):
        """Close position and update portfolio"""
        
        if symbol not in self.open_positions:
            self.logger.warning(f"No position found for {symbol}")
            return None
            
        position = self.open_positions[symbol]
        
        # Calculate P&L
        entry_price = position.get('entry_price', exit_price)
        position_size = position.get('size', 0)
        
        pnl = position_size * (exit_price - entry_price) / entry_price
        
        # Update portfolio
        self.portfolio_value += pnl
        self.cash_balance += position_size + pnl
        
        # Remove from open positions
        closed_position = self.open_positions.pop(symbol)
        closed_position.update({
            'status': 'closed',
            'exit_price': exit_price,
            'pnl': pnl,
            'close_timestamp': datetime.now()
        })
        
        self.logger.info(f"Position closed for {symbol}: P&L = {pnl:.2f}")
        return closed_position
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate current portfolio metrics"""
        
        total_open_pnl = 0
        total_unrealized_pnl = 0
        for symbol, position in self.open_positions.items():
            if position.get('status') == 'open':
                # Simple current P&L calculation (would need current price in real implementation)
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', entry_price)
                position_size = position.get('position_size', 0)
                
                # Calculate P&L based on position type
                if position.get('side') == 'buy':
                    unrealized_pnl = (current_price - entry_price) * position_size
                else:  # sell
                    unrealized_pnl = (entry_price - current_price) * position_size
                
                total_open_pnl += unrealized_pnl
                total_unrealized_pnl += unrealized_pnl
        
        # Calculate max drawdown (simplified)
        current_value = self.portfolio_value + total_open_pnl
        if not hasattr(self, '_peak_value') or current_value > self._peak_value:
            self._peak_value = current_value
        
        max_drawdown = (self._peak_value - current_value) / self._peak_value if self._peak_value > 0 else 0
        
        return {
            'total_value': current_value,
            'total_pnl': total_open_pnl,  # Add missing total_pnl key
            'cash_balance': self.cash_balance,
            'open_positions': len([p for p in self.open_positions.values() if p.get('status') == 'open']),
            'total_exposure': self._calculate_portfolio_exposure(),
            'unrealized_pnl': total_unrealized_pnl,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_portfolio_exposure(self) -> float:
        """Calculate current portfolio exposure"""
        
        total_exposure = 0
        for position in self.open_positions.values():
            if position.get('status') == 'open':
                total_exposure += position.get('size', 0)
        
        return total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def get_position_status(self, symbol: str) -> Dict[str, Any]:
        """Get current status of a position"""
        
        if symbol in self.open_positions:
            return self.open_positions[symbol]
        return None

class RealTimeMonitor:
    """Giám sát real-time các positions và tự động đóng SL/TP"""
    
    def __init__(self, risk_manager: AdvancedRiskManager):
        self.risk_manager = risk_manager
        self.discord_manager = DiscordNotificationManager()
        self.logger = LOG_MANAGER.get_logger('RiskManager')
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Bắt đầu giám sát real-time"""
        
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Dừng giám sát"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Real-time monitoring stopped")
    
    def _monitor_loop(self):
        """Vòng lặp giám sát chính"""
        
        while self.monitoring_active:
            try:
                # Monitor positions every 30 seconds
                time.sleep(30)
                
                if not self.risk_manager.open_positions:
                    continue
                
                # Check each open position
                positions_to_close = []
                
                for symbol, position in self.risk_manager.open_positions.items():
                    if position['status'] != 'open':
                        continue
                    
                    current_price = position['current_price']
                    stop_loss = position.get('stop_loss')
                    take_profit = position.get('take_profit')
                    
                    # Check stop loss and take profit
                    pnl = position['size'] * (current_price - position['entry_price'])
                    
                    # SL/TP logic
                    should_close = False
                    close_reason = ""
                    
                    if stop_loss:
                        if position['size'] > 0 and current_price <= stop_loss:  # Long position hit SL
                            should_close = True
                            close_reason = "Stop Loss"
                        elif position['size'] < 0 and current_price >= stop_loss:  # Short position hit SL
                            should_close = True
                            close_reason = "Stop Loss"
                    
                    if take_profit and not should_close:
                        if position['size'] > 0 and current_price >= take_profit:  # Long position hit TP
                            should_close = True
                            close_reason = "Take Profit"
                        elif position['size'] < 0 and current_price <= take_profit:  # Short position hit TP
                            should_close = True
                            close_reason = "Take Profit"
                    
                    # Weekend closure for non-crypto
                    if symbol != 'BTCUSD' and self._is_weekend_current_close():
                        should_close = True
                        close_reason = "Weekend Auto Close"
                    
                    if should_close:
                        positions_to_close.append({
                            'symbol': symbol,
                            'current_pnl': pnl,
                            'return_percentage': (current_price / position['entry_price'] - 1) * 100,
                            'reason': close_reason
                        })
                
                # Execute closures
                for close_info in positions_to_close:
                    self._close_position(close_info)
                
            except Exception as e:
                self.logger.error(f"Lỗi trong monitoring loop: {e}")
    
    def _close_position(self, close_info: Dict[str, Any]):
        """Đóng position và gửi notification"""
        
        symbol = close_info['symbol']
        pnl = close_info['current_pnl']
        return_pct = close_info['return_percentage']
        reason = close_info['reason']
        
        try:
            # Update position status
            position = self.risk_manager.open_positions[symbol]
            position['status'] = 'closed'
            position['pnl'] = pnl
            position['exit_time'] = datetime.now()
            
            # Calculate exit price from P&L
            entry_price = position.get('entry_price', 0)
            position_size = position.get('size', 0)
            exit_price = None
            
            if position_size != 0 and entry_price != 0:
                # exit_price = entry_price + (pnl / position_size)
                exit_price = entry_price + (pnl / abs(position_size)) * (1 if position_size > 0 else -1)
                position['exit_price'] = exit_price
            
            # Save to database
            import sqlite3
            
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            # Use database ID if available, otherwise fallback to symbol lookup
            position_id = position.get('db_id')
            
            if position_id:
                cursor.execute("""
                    UPDATE positions 
                    SET status='closed', exit_time=?, pnl=?, exit_price=? 
                    WHERE id=?
                """, (datetime.now().isoformat(), pnl, exit_price, position_id))
                
                self.logger.info(f"✅ Position updated in database using ID {position_id}")
            else:
                # Fallback to symbol-based update for backward compatibility
                cursor.execute("""
                    UPDATE positions 
                    SET status='closed', exit_time=?, pnl=?, exit_price=? 
                    WHERE symbol=? AND status='open'
                """, (datetime.now().isoformat(), pnl, exit_price, symbol))
                
                # Get the ID of the updated record for future reference
                cursor.execute("SELECT id FROM positions WHERE symbol=? AND status='closed' ORDER BY id DESC LIMIT 1", (symbol,))
                result = cursor.fetchone()
                if result:
                    position['db_id'] = result[0]
                
                self.logger.info(f"✅ Position updated in database using symbol {symbol}")
            
            conn.commit()
            conn.close()
            
            # Send Discord notification
            asyncio.run(self.discord_manager.send_position_closed(symbol, pnl, return_pct))
            
            self.logger.info(f"Position closed: {symbol} - PnL: ${pnl:.2f} ({return_pct:.2f}%) - Reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"Lỗi đóng position {symbol}: {e}")
    
    def _is_weekend_current_close(self) -> bool:
        """Kiểm tra có đóng toàn bộ positions vào cuối tuần không"""
        
        now = datetime.now()
        # Đóng positions vào thứ 6 sau 22:00 UTC
        if now.weekday() == 4 and now.hour >= 22:  # Friday 22:00+
            return True
        return False

# ===== MAIN TRADING BOT CONTROLLER =====
class TradingBotController:
    """Controller chính điều phối toàn bộ hệ thống bot"""
    
    def __init__(self):
        self.logger = LOG_MANAGER.get_logger('BotCore')
        self.running = False
        
        # Initialize core components
        self.api_manager = None
        self.data_manager = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.news_manager = None
        
        # ML Models
        self.ensemble_model = EnsembleModel()
        self.lstm_model = LSTMModel()
        self.auto_retrain_manager = None
        self.rl_agent = None
        
        # Trading Components
        self.trend_agent = TrendSpecialistAgent()
        self.news_agent = None
        self.risk_agent = RiskSpecialistAgent()
        self.master_agent = None
        self.risk_manager = None
        self.real_time_monitor = None
        
        # Observability
        self.discord_manager = DiscordNotificationManager()
        
        # Performance tracking
        self.cycle_count = 0
        self.last_data_fetch = None
        self.trading_session_start = None
    
    def setup_database(self):
        """Khởi tạo SQLite database và tạo bảng positions nếu chưa tồn tại"""
        
        try:
            import sqlite3
            from datetime import datetime
            
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            # Tạo bảng positions với cấu trúc hoàn chỉnh
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    position_type TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL,
                    risk_percentage REAL
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Database initialized at {Config.DB_PATH}")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def initialize(self):
        """Khởi tạo tất cả components"""
        
        try:
            self.logger.info("🚀 Initializing Trading Bot...")
            
            # Setup database first
            self.setup_database()
            
            # Initialize API Manager
            self.api_manager = APIManager()
            
            # Initialize Data Manager
            self.data_manager = EnhancedDataManager(self.api_manager)
            
            # Initialize News Manager
            self.news_manager = NewsEconomicManager(self.api_manager)
            
            # Initialize specialist agents
            self.news_agent = NewsSpecialistAgent(self.news_manager)
            
            # Initialize Master Agent với AI models
            self.master_agent = MasterAgent(
                self.trend_agent, 
                self.news_agent, 
                self.risk_agent,
                ensemble_model=self.ensemble_model,
                lstm_model=self.lstm_model,
                rl_agent=self.rl_agent
            )
            
            # Initialize Risk Manager
            self.risk_manager = AdvancedRiskManager(self.api_manager)
            
            # Initialize Auto Retrain Manager
            self.auto_retrain_manager = AutoRetrainManager(self.ensemble_model, self.lstm_model)
            
            # Initialize RL Agent
            import gymnasium as gym
            from stable_baselines3 import PPO
            
            # Create simplified trading environment for RL
            class SimpleTradingEnv(gym.Env):
                def __init__(self, data_manager):
                    super().__init__()
                    self.data_manager = data_manager
                    self.action_space = gym.spaces.Discrete(3)  # BUY, SELL, HOLD (0=BV, 1=SELL, 2=HOLD)
                    self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
                    self.current_step = 0
                    self.current_symbol = Config.SYMBOLS[0] if Config.SYMBOLS else "EURUSD"
                    
                def step(self, action):
                    # Get real features from data manager
                    obs = self.data_manager.get_features(self.current_symbol, 'H1')
                    
                    if obs is None:
                        # Fallback if no features available
                        obs = np.array([0.0] * 10, dtype=np.float32)
                    else:
                        # Ensure we have exactly 10 features
                        if len(obs) != 10:
                            obs = np.pad(obs, (0, 10 - len(obs)), 'constant')[:10]
                    
                    # Simplified reward calculation based on action
                    reward = self._calculate_reward(action)
                    
                    self.current_step += 1
                    done = False
                    info = {
                        'symbol': self.current_symbol,
                        'action': action,
                        'step': self.current_step
                    }
                    
                    return obs, reward, done, info
                
                def reset(self):
                    self.current_step = 0
                    # Try to get features for the current symbol
                    obs = self.data_manager.get_features(self.current_symbol, 'H1')
                    
                    if obs is None:
                        # Fallback if no features available
                        obs = np.array([0.0] * 10, dtype=np.float32)
                    else:
                        # Ensure we have exactly 10 features
                        if len(obs) != 10:
                            obs = np.pad(obs, (0, 10 - len(obs)), 'constant')[:10]
                    
                    return obs
                
                def _calculate_reward(self, action: int) -> float:
                    """Calculate reward based on action taken"""
                    # For now, return a small random reward
                    # In a real implementation, this would calculate based on actual trading performance
                    import random
                    reward_ranges = {0: (-0.1, 0.5), 1: (-0.1, 0.5), 2: (-0.05, 0.05)}  # BUY, SELL, HOLD
                    min_reward, max_reward = reward_ranges[action]
                    return random.uniform(min_reward, max_reward)
            
            if hasattr(self.data_manager, 'get_features'):
                trading_env = SimpleTradingEnv(self.data_manager)
                self.rl_agent = RLAgent(trading_env)
                self.rl_agent.create_model()
                self.logger.info("🤖 RL Agent initialized successfully")
            else:
                self.rl_agent = None
                self.logger.warning("RL Agent skipped - data_manager missing features")
            
            # Initialize Real-time Monitor
            self.real_time_monitor = RealTimeMonitor(self.risk_manager)
            
            self.logger.info("✅ Trading Bot initialized successfully")
            
            # Send startup notification
            await self._send_startup_notification()
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khởi tạo Trading Bot: {e}")
            await self.discord_manager.send_error_notification("Initialization Error", str(e))
            raise
    
    async def _send_startup_notification(self):
        """Gửi thông báo khởi động bot"""
        
        embed_data = {
            "title": "🤖 Trading Bot Started",
            "description": "AI/ML Trading Bot is now active and monitoring markets",
            "color": 0x00ff00,
            "fields": [
                {"name": "Symbols", "value": ", ".join(Config.SYMBOLS), "inline": True},
                {"name": "Portfolio Value", "value": "$100,000", "inline": True},
                {"name": "Market Regime", "value": "Monitoring...", "inline": True}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Trading Bot System"}
        }
        
        await self.discord_manager._send_embed(embed_data)
    
    async def run_trading_cycle(self):
        """Thực hiện một chu kỳ giao dịch hoàn chỉnh"""
        
        cycle_start_time = datetime.now()
        
        try:
            self.logger.info(f"📈 Starting trading cycle #{self.cycle_count + 1}")
            
            # 1. Fetch fresh market data
            market_data = await self._fetch_all_market_data()
            if not market_data:
                self.logger.warning("No market data available, skipping cycle")
                return
            
            # 2. Engineer features
            enriched_data = {}
            for symbol, timeframes in market_data.items():
                # Combine all timeframes into a single DataFrame for feature engineering
                combined_df = None
                for timeframe, df in timeframes.items():
                    if df is not None and hasattr(df, 'empty') and not df.empty:
                        # Add timeframe suffix to columns to avoid conflicts
                        df_with_timeframe = df.copy()
                        df_with_timeframe.columns = [f"{col}_{timeframe}" if col not in ['datetime', 'timestamp'] else col for col in df.columns]
                        
                        if combined_df is None:
                            combined_df = df_with_timeframe.copy()
                        else:
                            # Temporal alignment: align on datetime/timestamp
                            combined_df = pd.merge(combined_df, df_with_timeframe, on='datetime', how='outer', suffixes=('', f'_{timeframe}'))
                
                if combined_df is not None and not combined_df.empty:
                    enriched_data[symbol] = self.feature_engineer.engineer_all_features(combined_df, symbol)
            
            # 3. Analyze and decide
            portfolio_status = self._get_portfolio_status()
            decision = await self.master_agent.analyze_and_decide(enriched_data, portfolio_status)
            
            # 4. Execute trades if signals are strong enough
            if decision.get('signal_strength', 0) > 0.6 and decision.get('confidence', 0) > 0.5:
                await self._execute_trading_decisions(decision, enriched_data)
            
            # 5. Update models with new data
            await self._update_models_with_new_data(enriched_data)
            
            # 6. Check for retrain conditions
            if self.cycle_count % 24 == 0:  # Every 24 cycles
                await self._check_retrain_conditions(market_data)
            
            # 7. Calculate and report performance
            if self.cycle_count % 12 == 0:  # Every 12 cycles (hourly)
                await self._report_performance()
            
            self.cycle_count += 1
            
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            self.logger.info(f"✅ Trading cycle #{self.cycle_count} completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi trong trading cycle: {e}")
            await self.discord_manager.send_error_notification("Trading Cycle Error", str(e))
    
    async def _fetch_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """Lấy dữ liệu thị trường cho tất cả symbols"""
        
        market_data = {}
        
        async with self.api_manager:
            # Fetch data sequentially to avoid rate limiting
            for symbol in Config.SYMBOLS:
                symbol_data = {}
                for timeframe in Config.TIME_FRAMES:
                    try:
                        # Add delay between requests to respect rate limits
                        await asyncio.sleep(0.5)  # Increased from 0.1s to 0.5s
                        df = await self.data_manager.get_fresh_data(symbol, timeframe)
                        if df is not None:
                            symbol_data[timeframe] = df
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
                        continue
                
                if symbol_data:
                    market_data[symbol] = symbol_data
            
            self.last_data_fetch = datetime.now()
            self.logger.info(f"📊 Market data fetched for {len(market_data)} symbols")
        
        return market_data
    
    def _get_portfolio_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hiện tại của portfolio"""
        
        portfolio_metrics = self.risk_manager.calculate_portfolio_metrics()
        
        return {
            'total_value': portfolio_metrics['total_value'],
            'daily_pnl': portfolio_metrics['total_pnl'],
            'total_exposure': self.risk_manager._calculate_portfolio_exposure(),
            'max_drawdown': portfolio_metrics['max_drawdown'],
            'positions_count': len([p for p in self.risk_manager.open_positions.values() if p['status'] == 'open'])
        }
    
    async def _execute_trading_decisions(self, decision: Dict[str, Any], market_data: Dict[str, pd.DataFrame]):
        """Thực hiện các quyết định giao dịch"""
        
        action_type = decision.get('action_type', 'hold')
        
        if action_type == 'hold':
            return
        
        # Handle position sizing và execution
        for symbol, df in market_data.items():
            if df.empty:
                continue
            
            # Get current price và ATR
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
            
            # Calculate position size
            position_info = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=decision.get('signal_strength', 0),
                confidence=decision.get('confidence', 0),
                atr=atr,
                sentiment_score=decision.get('sentiment_score', 0)
            )
            
            # Validate trade
            validation = self.risk_manager.validate_trade(
                symbol=symbol,
                position_size=position_info['suggested_size'],
                position_type=action_type,
                current_price=current_price
            )
            
            if validation['allowed']:
                # Execute trade
                position_data = {
                    'size': position_info['suggested_size'] if action_type == 'buy' else -position_info['suggested_size'],
                    'entry_price': current_price,
                  'current_price': current_price,
                    'stop_loss': current_price - position_info['stop_loss_distance'] if action_type == 'buy' else current_price + position_info['stop_loss_distance'],
                    'take_profit': current_price + position_info['take_profit_distance'] if action_type == 'buy' else current_price - position_info['take_profit_distance'],
                    'position_type': action_type,
                    'risk_percentage': position_info['risk_percentage']
                }
                
                # Update position trong risk manager
                self.risk_manager.update_position(symbol, position_data)
                
                # Send Discord notification
                await self.discord_manager.send_position_opened(symbol, position_data)
                
                self.logger.info(f"✅ Trade executed: {action_type.upper()} {symbol} @ {current_price:.4f}")
                
            else:
                reasons = validation.get('reasons', [])
                self.logger.warning(f"🚫 Trade rejected for {symbol}: {', '.join(reasons)}")
    
    async def _initial_model_training(self):
        """Perform initial training for AI models if they're not trained yet"""
        
        try:
            if not self.running:
                return
                
            self.logger.info("🎓 Checking AI model training status...")
            
            # Check if models need initial training
            ensemble_trained = (self.ensemble_model and 
                             hasattr(self.ensemble_model, 'models') and 
                             self.ensemble_model.models and 
                             hasattr(self.ensemble_model, 'meta_model') and 
                             self.ensemble_model.meta_model)
            
            lstm_trained = (self.lstm_model and 
                          hasattr(self.lstm_model, 'is_trained') and 
                          self.lstm_model.is_trained)
            
            if ensemble_trained and lstm_trained:
                self.logger.info("✅ AI models are already trained!")
                return
            
            # Fetch historical data for initial training
            self.logger.info("📊 Fetching historical data for initial training...")
            historical_data = await self._fetch_initial_training_data()
            
            if not historical_data:
                self.logger.warning("⚠️ No historical data available for initial training")
                return
            
            # Train ensemble model if needed
            if not ensemble_trained:
                self.logger.info("🤖 Training ensemble model...")
                try:
                    for symbol, df in historical_data.items():
                        if len(df) > 100:  # Need sufficient data
                            # Prepare features and targets
                            features_df = df.select_dtypes(include=[np.number]).dropna()
                            if len(features_df) > 50:
                                targets = self._create_training_targets(df)
                                if targets is not None:
                                    await self.master_agent.trigger_ai_training(symbol, df)
                                    break  # Train on one symbol initially
                except Exception as e:
                    self.logger.warning(f"Ensemble model training failed: {e}")
            
            # Train LSTM model if needed
            if not lstm_trained:
                self.logger.info("🧠 Training LSTM model...")
                try:
                    for symbol, df in historical_data.items():
                        if len(df) > 100:  # Need sufficient data for LSTM
                            features_df = df.select_dtypes(include=[np.number]).dropna()
                            if len(features_df) > 50:
                                targets = self._create_training_targets(df)
                                if targets is not None:
                                    # Train LSTM model
                                    training_result = self.lstm_model.train(features_df, targets)
                                    if training_result.get('training_completed', False):
                                        self.logger.info("✅ LSTM model trained successfully!")
                                        break
                except Exception as e:
                    self.logger.warning(f"LSTM model training failed: {e}")
            
            self.logger.info("🎓 Initial model training completed!")
            
        except Exception as e:
            self.logger.error(f"❌ Initial model training failed: {e}")
    
    async def _fetch_initial_training_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for initial model training"""
        
        try:
            historical_data = {}
            
            # Use up to 2 symbols for initial training
            symbols = Config.SYMBOLS[:2]
            
            async with self.api_manager as api_mgr:
                for symbol in symbols:
                    try:
                        # Fetch data for multiple timeframes
                        df = await self.data_manager.get_fresh_data(symbol, 'H1')
                        if df is not None and len(df) > 50:
                            historical_data[symbol] = df
                            self.logger.info(f"📈 Fetched {len(df)} historical records for {symbol}")
                    except Exception as e:
                        self.logger.warning(f"Could not fetch training data for {symbol}: {e}")
                        continue
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching initial training data: {e}")
            return {}
    
    def _create_training_targets(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Create training targets from price data"""
        
        try:
            if len(df) < 2:
                return None
                
            # Simple target: 1 if next price is higher, 0 otherwise
            targets = []
            for i in range(len(df) - 1):
                current_price = df['close'].iloc[i]
                next_price = df['close'].iloc[i + 1]
                targets.append(1 if next_price > current_price else 0)
            
            return pd.Series(targets, name='target')
            
        except Exception:
            return None
    
    async def _update_models_with_new_data(self, enriched_data: Dict[str, pd.DataFrame]):
        """Cập nhật models với dữ liệu mới"""
        
        try:
            # Update online learning models
            for symbol, df in enriched_data.items():
                # Make sure df is a DataFrame, not a dict
                if isinstance(df, dict):
                    # Skip dict entries - they should be timeframes
                    continue
                    
                if hasattr(df, 'empty') and not df.empty and len(df) > Config.FEATURE_WINDOW:
                    # Get latest features
                    latest_row = df.iloc[-1:].select_dtypes(include=[np.number])
                    latest_features = latest_row.values[0] if not latest_row.empty else np.array([])
                    
                    # Create target (simplified: +1 if price went up next period, 0 otherwise)
                    if len(df) > 1:
                        current_price = df['close'].iloc[-1]
                        next_price = df['close'].iloc[-2]
                        target = 1 if current_price > next_price else 0
                        
                        self.auto_retrain_manager.online_manager.update_model(symbol, latest_features, float(target))
            
            self.logger.info("📈 Models updated with new data")
            
        except Exception as e:
            self.logger.error(f"Lỗi cập nhật models: {e}")
    
    async def _check_retrain_conditions(self, market_data: Dict[str, pd.DataFrame]):
        """Kiểm tra điều kiện retrain"""
        
        try:
            # Combine all market data for drift detection
            combined_data = pd.DataFrame()
            
            for symbol, df in market_data.items():
                if not df.empty:
                    symbol_data = df.copy()
                    symbol_data['symbol'] = symbol
                    combined_data = pd.concat([combined_data, symbol_data], ignore_index=True)
            
            if not combined_data.empty:
                retrain_conditions = self.auto_retrain_manager.check_retrain_conditions(combined_data)
                
                if retrain_conditions['should_retrain']:
                    self.logger.info("🔄 Retrain conditions met, preparing retrain..")
                    
                    # Simplified retrain với sample data
                    sample_data = combined_data.select_dtypes(include=[np.number]).dropna()
                    if len(sample_data) > 100:
                        X_sample = sample_data.iloc[:-1]
                        y_sample = pd.Series([1 if sample_data.iloc[i+1]['close'] > sample_data.head(i+1)['close'].mean() else 0 
                                            for i in range(len(X_sample))], index=X_sample.index)
                        
                        retrain_results = self.auto_retrain_manager.execute_retrain(X_sample, y_sample)
                        
                        if retrain_results['retrain_success']:
                            self.logger.info(f"✅ Retrain completed - Improvement: {retrain_results['improvement']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra retrain conditions: {e}")
    
    async def _report_performance(self):
        """Báo cáo hiệu suất định kỳ"""
        
        try:
            # Calculate portfolio metrics
            portfolio_metrics = self.risk_manager.calculate_portfolio_metrics()
            
            # Get model diagnostics
            model_diagnostics = self.auto_retrain_manager.get_model_diagnostics()
            
            # Send Discord report
            await self.discord_manager.send_performance_report(portfolio_metrics, model_diagnostics)
            
            self.logger.info("📊 Performance report sent")
            
        except Exception as e:
            self.logger.error(f"Lỗi gửi báo cáo performance: {e}")
    
    async def start(self, cycle_interval_minutes: int = 60):
        """Khởi động bot và bắt đầu trading loop"""
        
        if self.running:
            self.logger.warning("Bot is already running")
            return
        
        self.running = True
        self.trading_session_start = datetime.now()
        
        self.logger.info(f"🚀 Starting Trading Bot (cycle interval: {cycle_interval_minutes} minutes)")
        
        try:
            # Initialize all components
            await self.initialize()
            
            # Perform initial training if models are not trained
            await self._initial_model_training()
            
            # Start real-time monitoring
            if self.real_time_monitor:
                self.real_time_monitor.start_monitoring()
            
            # Main trading loop
            while self.running:
                try:
                    await self.run_trading_cycle()
                    
                    # Wait for next cycle
                    await asyncio.sleep(cycle_interval_minutes * 60)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error in trading loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
            
        except Exception as e:
            self.logger.error(f"Critical error starting bot: {e}")
            await self.discord_manager.send_error_notification("Critical Startup Error", str(e))
            raise
        
        finally:
            await self.stop()
    
    async def stop(self):
        """Dừng bot và cleanup"""
        
        self.logger.info("🛑 Stopping Trading Bot...")
        self.running = False
        
        try:
            # Stop real-time monitoring
            if self.real_time_monitor:
                self.real_time_monitor.stop_monitoring()
            
            # Close all connections
            if self.api_manager:
                await self.api_manager.__aexit__(None, None, None)
            
            # Send shutdown notification
            embed_data = {
                "title": "⏹️ Trading Bot Stopped",
                "description": f"Trading Bot has been shut down after {self.cycle_count} cycles",
                "color": 0xff8800,
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "Session Ended"}
            }
            
            await self.discord_manager._send_embed(embed_data)
            
            self.logger.info("✅ Trading Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Lỗi trong stop process: {e}")

# ===== MAIN EXECUTION =====
async def main():
    """Hàm main của chương trình"""
    
    try:
        # Create và start trading bot
        trading_bot = TradingBotController()
        
        # Start bot với cycle interval 60 minutes
        await trading_bot.start(cycle_interval_minutes=60)
        
    except Exception as e:
        logger = LOG_MANAGER.get_logger('BotCore')
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")
        
        # Send emergency notification
        try:
            discord_manager = DiscordNotificationManager()
            await discord_manager.send_error_notification("Fatal System Error", str(e))
        except:
            pass  # Don't fail if Discord notification fails

if __name__ == "__main__":
    """
    Trading Bot AI/ML Startup
    ========================
    
    Tính năng chính:
    - Ensemble ML models với AutoML optimization
    - LSTM với Attention mechanism
    - Reinforcement Learning với PPO
    - Multi-specialist agent architecture
    - Advanced risk management
    - Real-time monitoring và auto SL/TP
    - News sentiment analysis với AI
    - Concept drift detection và auto retrain
    - Discord notifications và monitoring
    """
    
    print("🚀 Starting AI/ML Trading Bot...")
    print("=" * 50)
    
    try:
        # Run main function
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
        print("Shutting down gracefully...")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    finally:
        print("👋 Trading Bot stopped")
        print("Thank you for using AI/ML Trading Bot v2.0")