import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from textblob import TextBlob
import yfinance as yf
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedHMMTrader:
    def __init__(self, n_states=3, leverage_limit=2.0, stop_loss=-0.02, 
                 take_profit=0.03, risk_per_trade=0.02, optimization_window=126):
        self.n_states = n_states
        self.leverage_limit = leverage_limit
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_per_trade = risk_per_trade
        self.optimization_window = optimization_window
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize HMM and other models"""
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.regime_classifier = KMeans(n_clusters=4, random_state=42)
        
    class AdaptiveParameters:
        """Nested class for parameter optimization"""
        def __init__(self):
            self.optimal_params = {
                'stop_loss': -0.02,
                'take_profit': 0.03,
                'risk_per_trade': 0.02,
                'leverage_limit': 2.0,
                'confidence_threshold': 0.6
            }
            
        def optimize(self, historical_data, returns):
            """Optimize parameters based on historical performance"""
            def objective(params):
                sl, tp, rpt, lev, conf = params
                # Calculate Sharpe ratio with penalties for risk
                sharpe = np.sqrt(252) * returns.mean() / returns.std()
                max_drawdown = (1 + returns).cumprod().div(
                    (1 + returns).cumprod().cummax()
                ).min()
                
                # Penalize for excessive risk
                risk_penalty = (lev - 1) * 0.1 + (abs(sl) > 0.05) * 0.2
                return -(sharpe * (1 + max_drawdown) - risk_penalty)
            
            # Define bounds for parameters
            bounds = (
                (-0.05, -0.01),  # stop_loss
                (0.02, 0.06),    # take_profit
                (0.01, 0.05),    # risk_per_trade
                (1.0, 3.0),      # leverage_limit
                (0.5, 0.8)       # confidence_threshold
            )
            
            result = minimize(
                objective,
                x0=[self.optimal_params[k] for k in self.optimal_params],
                bounds=bounds,
                method='SLSQP'
            )
            
            if result.success:
                self.optimal_params = {
                    'stop_loss': result.x[0],
                    'take_profit': result.x[1],
                    'risk_per_trade': result.x[2],
                    'leverage_limit': result.x[3],
                    'confidence_threshold': result.x[4]
                }
            
            return self.optimal_params
    
    def detect_market_regime(self, df):
        """Enhanced market regime detection"""
        # Calculate regime features
        regime_features = pd.DataFrame()
        
        # Trend features
        regime_features['trend'] = df['close'].rolling(20).mean().pct_change()
        regime_features['trend_strength'] = df['ADX']
        
        # Volatility features
        regime_features['volatility'] = df['returns'].rolling(20).std()
        regime_features['volatility_change'] = regime_features['volatility'].pct_change()
        
        # Volume features
        if 'volume' in df.columns:
            regime_features['volume_trend'] = df['volume'].rolling(20).mean().pct_change()
            regime_features['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Momentum features
        regime_features['momentum'] = df['returns'].rolling(10).mean()
        regime_features['momentum_change'] = regime_features['momentum'].pct_change()
        
        # Clean and scale features
        regime_features = regime_features.dropna()
        scaled_features = StandardScaler().fit_transform(regime_features)
        
        # Classify regime
        regimes = self.regime_classifier.fit_predict(scaled_features)
        
        # Determine regime characteristics
        regime_characteristics = {
            'trend_regime': np.sign(regime_features['trend'].mean()),
            'volatility_regime': 'high' if regime_features['volatility'].mean() > regime_features['volatility'].quantile(0.75) else 'low',
            'momentum_regime': 'positive' if regime_features['momentum'].mean() > 0 else 'negative',
            'regime_cluster': regimes[-1]
        }
        
        return regime_characteristics
    
    def analyze_sentiment(self, symbol, days=30):
        """Analyze market sentiment using news and social media"""
        sentiment_scores = []
        
        try:
            # Get news headlines from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for article in news:
                if article['type'] == 'STORY':
                    # Analyze sentiment of headline and summary
                    headline_blob = TextBlob(article['title'])
                    summary_blob = TextBlob(article['summary'])
                    
                    sentiment_scores.append({
                        'date': pd.to_datetime(article['providerPublishTime'], unit='s'),
                        'headline_sentiment': headline_blob.sentiment.polarity,
                        'summary_sentiment': summary_blob.sentiment.polarity
                    })
            
            # Convert to DataFrame and calculate aggregate sentiment
            sentiment_df = pd.DataFrame(sentiment_scores)
            if not sentiment_df.empty:
                sentiment_df['composite_sentiment'] = (
                    sentiment_df['headline_sentiment'] * 0.6 +
                    sentiment_df['summary_sentiment'] * 0.4
                )
                
                return {
                    'current_sentiment': sentiment_df['composite_sentiment'].mean(),
                    'sentiment_trend': sentiment_df['composite_sentiment'].diff().mean(),
                    'sentiment_volatility': sentiment_df['composite_sentiment'].std()
                }
        
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
        
        return {'current_sentiment': 0, 'sentiment_trend': 0, 'sentiment_volatility': 0}
    
    def adaptive_position_sizing(self, capital, current_price, atr, regime_info, sentiment_info):
        """Advanced position sizing considering market regime and sentiment"""
        base_position = self.calculate_position_size(capital, current_price, atr)
        
        # Adjust for market regime
        regime_multiplier = 1.0
        if regime_info['volatility_regime'] == 'high':
            regime_multiplier *= 0.7
        if regime_info['trend_regime'] < 0:
            regime_multiplier *= 0.8
            
        # Adjust for sentiment
        sentiment_multiplier = 1.0 + (sentiment_info['current_sentiment'] * 0.2)
        
        # Apply constraints
        adjusted_position = base_position * regime_multiplier * sentiment_multiplier
        return min(adjusted_position, capital * self.leverage_limit / current_price)
    
    def generate_enhanced_signals(self, df, symbol):
        """Generate signals with regime detection and sentiment analysis"""
        features = self.prepare_features(df)
        states = self.model.predict(features)
        state_probs = self.model.predict_proba(features)
        
        signals = pd.DataFrame(index=df.index[len(df)-len(states):])
        signals['position'] = 0
        signals['confidence'] = 0
        signals['stop_loss'] = 0
        signals['take_profit'] = 0
        
        # Get market regime and sentiment information
        regime_info = self.detect_market_regime(df)
        sentiment_info = self.analyze_sentiment(symbol)
        
        for i in range(len(states)):
            current_state = states[i]
            confidence = np.max(state_probs[i])
            
            # Enhanced signal generation considering all factors
            signal = self.calculate_signal(
                df.iloc[i],
                current_state,
                confidence,
                regime_info,
                sentiment_info
            )
            
            signals['position'].iloc[i] = signal
            signals['confidence'].iloc[i] = confidence
            
            if signal != 0:
                # Dynamic risk management levels based on market conditions
                volatility_adjust = 1.0
                if regime_info['volatility_regime'] == 'high':
                    volatility_adjust = 1.2
                
                signals['stop_loss'].iloc[i] = df['close'].iloc[i] * (
                    1 - self.stop_loss * signal * volatility_adjust
                )
                signals['take_profit'].iloc[i] = df['close'].iloc[i] * (
                    1 + self.take_profit * signal / volatility_adjust
                )
        
        return signals
    
    def calculate_signal(self, data_row, state, confidence, regime_info, sentiment_info):
        """Calculate trading signal considering all factors"""
        signal = 0
        
        # Base signal from HMM state
        if state == 0 and confidence > 0.6:  # Bearish state
            signal = -1
        elif state == 2 and confidence > 0.6:  # Bullish state
            signal = 1
            
        # Apply regime filters
        if regime_info['volatility_regime'] == 'high' and abs(signal) > 0:
            signal *= 0.5  # Reduce position size in high volatility
            
        if regime_info['trend_regime'] * signal < 0:
            signal = 0  # Don't trade against the major trend
            
        # Apply sentiment filters
        if abs(signal) > 0:
            if signal > 0 and sentiment_info['current_sentiment'] < -0.5:
                signal = 0  # Don't go long in very negative sentiment
            elif signal < 0 and sentiment_info['current_sentiment'] > 0.5:
                signal = 0  # Don't go short in very positive sentiment
                
        return signal
    
    def adaptive_backtest(self, df, symbol, initial_capital=100000):
        """Enhanced backtesting with adaptive parameters"""
        adaptive_params = self.AdaptiveParameters()
        results_window = []
        
        # Split data into windows for optimization
        for i in range(0, len(df), self.optimization_window):
            window_df = df.iloc[i:i+self.optimization_window]
            if len(window_df) < 30:  # Skip if window is too small
                continue
                
            # Optimize parameters for this window
            window_returns = window_df['close'].pct_change()
            optimal_params = adaptive_params.optimize(window_df, window_returns)
            
            # Update trader parameters
            self.stop_loss = optimal_params['stop_loss']
            self.take_profit = optimal_params['take_profit']
            self.risk_per_trade = optimal_params['risk_per_trade']
            self.leverage_limit = optimal_params['leverage_limit']
            
            # Generate and execute trades
            signals = self.generate_enhanced_signals(window_df, symbol)
            window_results = self.execute_trades(window_df, signals, initial_capital)
            results_window.append(window_results)
            
            # Update initial capital for next window
            initial_capital = window_results['portfolio']['equity'].iloc[-1]
        
        # Combine results from all windows
        combined_results = self.combine_results(results_window)
        return combined_results
    
    def execute_trades(self, df, signals, initial_capital):
        """Execute trades with advanced risk management"""
        portfolio = self.initialize_portfolio(signals, df, initial_capital)
        
        current_position = 0
        entry_price = 0
        
        for i in range(1, len(portfolio)):
            # Risk management checks
            if current_position != 0:
                if self.check_stop_loss(portfolio, i, current_position, entry_price) or \
                   self.check_take_profit(portfolio, i, current_position, entry_price):
                    self.close_position(portfolio, i)
                    current_position = 0
            
            # New position entry
            if signals['position'][i] != 0 and current_position == 0:
                current_position = self.enter_position(
                    portfolio, signals, i, 
                    self.detect_market_regime(df.iloc[:i]),
                    self.analyze_sentiment(df.index[i].strftime('%Y-%m-%d'))
                )
                entry_price = portfolio['close'][i]
            
            # Update portfolio value
            self.update_portfolio_value(portfolio, i)
        
        return self.calculate_performance_metrics(portfolio)
    
    def combine_results(self, results_window):
        """Combine results from multiple trading windows"""
        combined_portfolio = pd.concat([r['portfolio'] for r in results_window])
        combined_returns = combined_portfolio['equity'].pct_change()
        
        return {
            'total_return': (combined_portfolio['equity'].iloc[-1] / combined_portfolio['equity'].iloc[0] - 1) * 100,
            'sharpe_ratio': np.sqrt(252) * combined_returns.mean() / combined_returns.std(),
            'max_drawdown': (combined_portfolio['equity'] / combined_portfolio['equity'].cummax() - 1).min() * 100,
            'win_rate': len(combined_returns[combined_returns > 0]) / len(combined_returns[combined_returns != 0]),
            'portfolio': combined_portfolio
        }

# Example usage
def run_advanced_strategy(data, symbol):
    trader = AdvancedHMMTrader(
        n_states=3,
        leverage_limit=2.0,
        stop_loss=0.02,
        take_profit=0.03,
        risk_per_trade=0.02,
        optimization_window=126
    )
    trader.train(data)
    results = trader.adaptive_backtest(data, symbol)
    return results