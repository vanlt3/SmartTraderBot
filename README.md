# 🤖 Trading Bot AI/ML Tiên tiến v2.0

Bot giao dịch tự động tích hợp các công nghệ AI/ML hiện đại với kiến trúc Master-Specialist.

## 🌟 Tính năng chính

### 🧠 AI/ML Models
- **Ensemble Models**: XGBoost, LightGBM, Random Forest với AutoML (Optuna)
- **LSTM với Attention**: Deep learning cho time series prediction
- **Reinforcement Learning**: PPO agent cho portfolio management
- **Concept Drift Detection**: Tự động phát hiện và retrain khi market thay đổi

### 📊 Technical Analysis nâng cao
- **Wyckoff Patterns**: Spring, Upthrust, Accumulation/Distribution
- **Supply/Demand Zones**: Tự động identification và distance calculation
- **Market Regime Detection**: Trending vs Sideways markets
- **RSI Divergence**: Advanced momentum analysis
- **Multi-timeframe**: M15, H1, H4, D1 data integration

### 📰 News & Sentiment Analysis
- **Multi-source news**: Finnhub, MarketAux, NewsAPI, EODHD
- **AI Sentiment Analysis**: Gemini 1.5 Flash API integration
- **Economic Calendar**: Integration với lịch kinh tế

### 🏗️ Kiến trúc Master-Specialist
- **Master Agent**: Điều phối và ra quyết định cuối cùng
- **Specialist Agents**: Trend, News, Risk, Sentiment, Volatility analysis
- **Consensus Mechanism**: Aggregated decision making

### ⚔️ Advanced Risk Management
- **Multi-layer Risk**: Portfolio và trade-level risk
- **Dynamic Position Sizing**: Based on confidence, ATR, sentiment
- **Correlation Monitoring**: Prevent correlated trades
- **Real-time SL/TP**: Automatic stop loss và take profit management

### 📡 Observability & Monitoring
- **Discord Notifications**: Rich embed notifications với webhook
- **Real-time Monitoring**: Automatic position management
- **Performance Tracking**: SQLite database với history
- **Colorful Logging**: Enhanced logging với emojis và colors

## 🚀 Cài đặt và Sử dụng

### 1. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Keys
File đã có sẵn các API keys trong class `Config`:
- Alpha Vantage: `FK3YQ1IKSC4E1AL5`
- Finnhub: `d1b3ichr01qjhvtsbj8g`
- MarketAux: `CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql`
- NewsAPI: `abd8f43b808f42fdb8d28fb1c429af72`
- EODHD: `68bafd7d44a7f0.25202650`
- OANDA: `814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d`

### 3. Chạy Bot
```bash
python3 trading_bot.py
```

## 📈 Symbols được hỗ trợ
- **XAUUSD** (Gold/USD)
- **EURUSD** (Euro/USD)
- **NAS100** (NASDAQ 100)
- **BTCUSD** (Bitcoin/USD)

## ⚙️ Cấu hình

### Trading Parameters
- `MAX_POSITION_SIZE`: 2% per trade
- `MAX_CORRELATION_THRESHOLD`: 0.7
- `STOP_LOSS_MULTIPLIER`: 2.0 ATR
- `TAKE_PROFIT_MULTIPLIER`: 3.0 ATR

### Cycle Settings
- Mặc định: 60 phút per cycle
- Real-time monitoring: Continuous
- Performance reports: Hourly
- Retrain checks: Every 24 cycles

## 🔧 Customization

### Thêm Symbols mới
```python
# Trong class Config
SYMBOLS = ["XAUUSD", "EURUSD", "NAS100", "BTCUSD", "YOUNSYMBOLS"]
```

### Điều chỉnh Risk Parameters
```python
# Trong class Config
MAX_POSITION_SIZE = 0.02  # 2% per trade
STOP_LOSS_MULTIPLIER = 2.5  # Adjust SL distance
```

## 📊 Features Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Manager | aiohttp | Async data fetching với rate limiting |
| Data Manager | pandas | Multi-timeframe data management |
| Feature Engineer | ta + custom | Advanced technical indicators |
| Ensemble Models | sklearn + optuna | AutoML optimization |
| LSTM Model | tensorflow | Deep learning predictions |
| RL Environment | gym + stable-baselines3 | Portfolio optimization |
| Risk Manager | sqlite3 | Position tracking và risk control |
| Discord Bot | webhooks | Rich notifications |

## 🎯 Architecture Benefits

1. **Scalable**: Module design cho easy extension
2. **Robust**: Comprehensive error handling và logging
3. **Adaptive**: Self-learning với concept drift detection
4. **Observable**: Rich monitoring và notifications
5. **Risk-Aware**: Advanced risk management system

## 📝 Notes

- Bot tự động đóng positions (trừ crypto) vào cuối tuần
- Real-time monitoring cho SL/TP management
- Auto-retrain khi detect performance decline
- Discord notifications cho all major events

---

**Version 2.0.0** - Advanced AI/ML Trading Bot với Master-Specialist Architecture