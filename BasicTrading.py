import numpy as np
from hmmlearn import hmm
import pandas as pd
from sklearn.preprocessing import StandardScaler

class HMMTrader:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, prices):
        """Calculate technical indicators as features"""
        df = pd.DataFrame(prices, columns=['close'])
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility (20-day rolling standard deviation)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate SMA crossovers
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['sma_crossover'] = df['SMA20'] - df['SMA50']
        
        # Prepare feature matrix
        features = ['returns', 'volatility', 'sma_crossover']
        feature_matrix = df[features].dropna()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        return scaled_features
    
    def train(self, prices):
        """Train the HMM model on historical price data"""
        features = self.prepare_features(prices)
        self.model.fit(features)
        return self
    
    def predict_state(self, features):
        """Predict the current market state"""
        return self.model.predict(features)
    
    def generate_signals(self, prices):
        """Generate trading signals based on predicted states"""
        features = self.prepare_features(prices)
        states = self.predict_state(features)
        
        # Calculate state probabilities
        state_probs = self.model.predict_proba(features)
        
        # Generate trading signals
        signals = np.zeros(len(states))
        for i in range(len(states)):
            current_state = states[i]
            
            # Example trading logic based on states and probabilities
            if current_state == 0:  # Bearish state
                signals[i] = -1
            elif current_state == 1:  # Neutral state
                signals[i] = 0
            else:  # Bullish state
                signals[i] = 1
                
            # Adjust signal based on prediction confidence
            confidence = np.max(state_probs[i])
            if confidence < 0.6:  # Low confidence threshold
                signals[i] = 0
                
        return signals
    
    def backtest(self, prices, initial_capital=100000):
        """Perform backtesting of the strategy"""
        signals = self.generate_signals(prices)
        
        # Calculate positions and returns
        positions = pd.Series(signals, index=prices.index[100:])  # Offset for feature calculation
        returns = pd.Series(prices).pct_change()
        strategy_returns = positions.shift(1) * returns[positions.index]
        
        # Calculate portfolio value
        portfolio_value = (1 + strategy_returns).cumprod() * initial_capital
        
        # Calculate performance metrics
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()
        
        results = {
            'portfolio_value': portfolio_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': (portfolio_value[-1] / initial_capital - 1) * 100
        }
        
        return results

# Example usage
def run_hmm_strategy(price_data):
    trader = HMMTrader(n_states=3)
    trader.train(price_data)
    results = trader.backtest(price_data)
    return results, trader.generate_signals(price_data)