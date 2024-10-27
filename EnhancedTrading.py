import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class EnhancedHMMTrader:
    def __init__(self, n_states=3, leverage_limit=2.0, stop_loss=-0.02, 
                 take_profit=0.03, risk_per_trade=0.02):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.leverage_limit = leverage_limit
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_per_trade = risk_per_trade
        
    def calculate_technical_features(self, df):
        """Calculate advanced technical indicators"""
        # Momentum indicators
        df['RSI'] = self._calculate_rsi(df['close'], periods=14)
        df['MACD'], df['MACD_signal'] = self._calculate_macd(df['close'])
        
        # Volatility indicators
        df['ATR'] = self._calculate_atr(df)
        df['BBands_upper'], df['BBands_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume-based indicators
        if 'volume' in df.columns:
            df['OBV'] = self._calculate_obv(df)
            df['VWAP'] = self._calculate_vwap(df)
        
        # Trend indicators
        df['ADX'] = self._calculate_adx(df)
        
        return df
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_atr(self, df, periods=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(periods).mean()
    
    def _calculate_bollinger_bands(self, prices, periods=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=periods).mean()
        std = prices.rolling(window=periods).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = df['volume'].copy()
        obv[df['close'] < df['close'].shift()] *= -1
        return obv.cumsum()
    
    def _calculate_vwap(self, df):
        """Calculate Volume-Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_adx(self, df, periods=14):
        """Calculate Average Directional Index"""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        tr = self._calculate_atr(df, 1)
        
        plus_di = 100 * (plus_dm.rolling(periods).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(periods).mean() / tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(periods).mean()
        return adx
    
    def prepare_features(self, df):
        """Prepare and combine all features"""
        # Calculate all technical indicators
        df = self.calculate_technical_features(df)
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Create feature matrix
        feature_columns = [
            'returns', 'volatility', 'RSI', 'MACD', 'ATR',
            'BBands_upper', 'BBands_lower', 'ADX'
        ]
        
        if 'volume' in df.columns:
            feature_columns.extend(['OBV', 'VWAP'])
            
        feature_matrix = df[feature_columns].dropna()
        return self.scaler.fit_transform(feature_matrix)
    
    def calculate_position_size(self, capital, current_price, atr):
        """Calculate position size based on ATR and risk per trade"""
        risk_amount = capital * self.risk_per_trade
        atr_multiple = 2  # Use 2x ATR for stop loss
        position_size = risk_amount / (atr * atr_multiple)
        return min(position_size, capital * self.leverage_limit / current_price)
    
    def generate_signals(self, df):
        """Generate sophisticated trading signals"""
        features = self.prepare_features(df)
        states = self.model.predict(features)
        state_probs = self.model.predict_proba(features)
        
        signals = pd.DataFrame(index=df.index[len(df)-len(states):])
        signals['position'] = 0
        signals['confidence'] = 0
        signals['stop_loss'] = 0
        signals['take_profit'] = 0
        
        for i in range(len(states)):
            current_state = states[i]
            confidence = np.max(state_probs[i])
            
            # Complex signal generation based on state and multiple indicators
            if current_state == 0:  # Bearish state
                if (df['RSI'].iloc[i] > 70 and 
                    df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and
                    confidence > 0.6):
                    signals['position'].iloc[i] = -1
                    
            elif current_state == 2:  # Bullish state
                if (df['RSI'].iloc[i] < 30 and 
                    df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and
                    confidence > 0.6):
                    signals['position'].iloc[i] = 1
            
            # Set confidence and risk management levels
            signals['confidence'].iloc[i] = confidence
            if signals['position'].iloc[i] != 0:
                signals['stop_loss'].iloc[i] = df['close'].iloc[i] * (1 - self.stop_loss * signals['position'].iloc[i])
                signals['take_profit'].iloc[i] = df['close'].iloc[i] * (1 + self.take_profit * signals['position'].iloc[i])
        
        return signals
    
    def backtest(self, df, initial_capital=100000):
        """Enhanced backtesting with risk management"""
        signals = self.generate_signals(df)
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['position'] = signals['position']
        portfolio['close'] = df['close'][signals.index]
        portfolio['returns'] = df['close'][signals.index].pct_change()
        portfolio['ATR'] = self._calculate_atr(df)[signals.index]
        
        # Initialize portfolio metrics
        portfolio['capital'] = initial_capital
        portfolio['shares'] = 0
        portfolio['equity'] = initial_capital
        
        current_position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(1, len(portfolio)):
            # Check stop loss and take profit for existing positions
            if current_position != 0:
                if (current_position > 0 and portfolio['close'][i] <= stop_loss) or \
                   (current_position < 0 and portfolio['close'][i] >= stop_loss):
                    # Stop loss hit
                    portfolio.loc[portfolio.index[i], 'shares'] = 0
                    portfolio.loc[portfolio.index[i], 'capital'] = portfolio['equity'][i-1]
                    current_position = 0
                    
                elif (current_position > 0 and portfolio['close'][i] >= take_profit) or \
                     (current_position < 0 and portfolio['close'][i] <= take_profit):
                    # Take profit hit
                    portfolio.loc[portfolio.index[i], 'shares'] = 0
                    portfolio.loc[portfolio.index[i], 'capital'] = portfolio['equity'][i-1]
                    current_position = 0
            
            # Check for new signals
            if portfolio['position'][i] != 0 and current_position == 0:
                # Calculate position size
                pos_size = self.calculate_position_size(
                    portfolio['capital'][i-1],
                    portfolio['close'][i],
                    portfolio['ATR'][i]
                )
                
                portfolio.loc[portfolio.index[i], 'shares'] = pos_size * portfolio['position'][i]
                current_position = portfolio['position'][i]
                entry_price = portfolio['close'][i]
                stop_loss = signals['stop_loss'][i]
                take_profit = signals['take_profit'][i]
            
            # Update portfolio value
            portfolio.loc[portfolio.index[i], 'equity'] = \
                portfolio['capital'][i] + portfolio['shares'][i] * portfolio['close'][i]
        
        # Calculate performance metrics
        portfolio['returns'] = portfolio['equity'].pct_change()
        
        results = {
            'total_return': (portfolio['equity'][-1] / initial_capital - 1) * 100,
            'sharpe_ratio': np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std(),
            'max_drawdown': (portfolio['equity'] / portfolio['equity'].cummax() - 1).min() * 100,
            'win_rate': len(portfolio[portfolio['returns'] > 0]) / len(portfolio[portfolio['returns'] != 0]),
            'portfolio': portfolio
        }
        
        return results

# Example usage
def run_enhanced_strategy(data):
    trader = EnhancedHMMTrader(
        n_states=3,
        leverage_limit=2.0,
        stop_loss=0.02,
        take_profit=0.03,
        risk_per_trade=0.02
    )
    trader.train(data)
    results = trader.backtest(data)
    return results