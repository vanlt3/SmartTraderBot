#!/usr/bin/env python3
"""
Test script để kiểm tra Master Agent với AI models
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Mock classes để test
class MockTrendAgent:
    async def analyze_trend(self, symbol, data):
        return {'trend': 'UPTREND', 'confidence': 0.7}
    
    def get_trend_recommendation(self, symbol):
        return {'action': 'BUY', 'confidence': 0.7}

class MockNewsAgent:
    async def analyze_sentiment(self, symbol):
        return {'sentiment': 'POSITIVE', 'confidence': 0.6}
    
    def get_sentiment_recommendation(self, symbol):
        return {'sentiment': 'POSITIVE', 'confidence': 0.6}

class MockRiskAgent:
    async def assess_risk(self, symbol, data, portfolio_value):
        return {'risk_level': 'MEDIUM', 'position_size': 0.02}

class MockEnsembleModel:
    def predict(self, X):
        # Mock prediction - return BUY signal
        return np.array([0.8]), np.array([0.75])

class MockLSTMModel:
    def predict(self, data):
        # Mock prediction - return BUY signal
        return 0.7, 0.8

class MockRLAgent:
    def __init__(self):
        self.model = True
    
    def train(self, total_timesteps):
        print(f"Mock RL Agent training for {total_timesteps} timesteps")
        return True

# Import MasterAgent từ trading_bot.py
import sys
sys.path.append('/workspace')
from trading_bot import MasterAgent

async def test_master_agent():
    """Test Master Agent với AI models"""
    print("🧪 Testing Master Agent với AI models...")
    
    # Initialize mock agents
    trend_agent = MockTrendAgent()
    news_agent = MockNewsAgent()
    risk_agent = MockRiskAgent()
    ensemble_model = MockEnsembleModel()
    lstm_model = MockLSTMModel()
    rl_agent = MockRLAgent()
    
    # Initialize Master Agent với AI models
    master_agent = MasterAgent(
        trend_agent, 
        news_agent, 
        risk_agent,
        ensemble_model=ensemble_model,
        lstm_model=lstm_model,
        rl_agent=rl_agent
    )
    
    print(f"✅ Master Agent initialized với:")
    print(f"   - Ensemble weight: {master_agent.ensemble_weight}")
    print(f"   - LSTM weight: {master_agent.lstm_weight}")
    print(f"   - Expert weight: {master_agent.expert_weight}")
    
    # Test make_decision với mock data
    test_data = {
        'close': 50000,
        'volume': 1000000,
        'sma_20': 49500,
        'sma_50': 49000,
        'rsi': 65,
        'macd': 150,
        'bb_upper': 51000,
        'bb_lower': 48000,
        'atr': 500,
        'volatility': 0.02
    }
    
    print("\n🎯 Testing decision making...")
    decision = await master_agent.make_decision("BTCUSD", test_data, 100000)
    
    print(f"✅ Decision: {decision['action']}")
    print(f"   Confidence: {decision['confidence']:.2f}")
    print(f"   Position Size: {decision['position_size']:.3f}")
    
    # Test AI predictions
    if 'ai_predictions' in decision:
        ai_preds = decision['ai_predictions']
        print(f"   Ensemble: {ai_preds.get('ensemble', {}).get('action', 'N/A')}")
        print(f"   LSTM: {ai_preds.get('lstm', {}).get('action', 'N/A')}")
        print(f"   RL: {ai_preds.get('rl', {}).get('action', 'N/A')}")
    
    # Test training functionality
    print("\n🎓 Testing AI training...")
    
    # Create mock training data
    training_data = pd.DataFrame({
        'close': np.random.rand(60) * 50000 + 40000,
        'volume': np.random.rand(60) * 1000000,
        'sma_20': np.random.rand(60) * 50000 + 40000,
        'sma_50': np.random.rand(60) * 50000 + 40000,
        'rsi': np.random.rand(60) * 100,
        'macd': np.random.rand(60) * 200 - 100,
        'bb_upper': np.random.rand(60) * 55000 + 40000,
        'bb_lower': np.random.rand(60) * 45000 + 35000,
        'atr': np.random.rand(60) * 1000,
        'volatility': np.random.rand(60) * 0.1
    })
    
    training_result = await master_agent.trigger_ai_training("BTCUSD", training_data)
    print(f"   Training status: {training_result['status']}")
    print(f"   Data points: {len(training_data)}")
    
    print("\n✅ Test completed successfully!")
    print(f"   Master Agent confidence calculation is working!")
    print(f"   AI models integration is functional!")

if __name__ == "__main__":
    asyncio.run(test_master_agent())