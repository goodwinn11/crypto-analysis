import ccxt
import pandas as pd
import numpy as np
import json
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class GPTAnalyzer:
    """
    GPT-based analysis for trading signals with detailed Russian format output.
    Generates comprehensive analysis matching the sravni.ae/crypto style.
    Updated to support GPT-5 (gpt-o1-preview) with proper parameters.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.enabled = bool(self.api_key)
        
        # Get model from environment or use GPT-5 as default
        self.model = model or os.getenv('GPT_MODEL', 'gpt-o1-preview')
        
        # Use GPT-4 for detailed Russian analysis if main model is GPT-5
        self.detail_model = 'gpt-4-turbo-preview' if self.model == 'gpt-o1-preview' else self.model
        
        if self.enabled:
            try:
                import openai
                self.openai = openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f'GPT Analyzer initialized with model: {self.model}')
                if self.model == 'gpt-o1-preview':
                    logger.info(f'Using GPT-5 for analysis with GPT-4 for detailed Russian content')
                else:
                    logger.info(f'Using {self.model} for all analysis')
            except ImportError:
                logger.warning('OpenAI library not installed. Run: pip install openai')
                self.enabled = False
            except Exception as e:
                logger.warning(f'Failed to initialize GPT: {e}')
                self.enabled = False
        else:
            logger.info('GPT Analyzer disabled (no API key provided)')
    
    def analyze_signal(self, signal_data: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Analyze a trading signal using GPT for detailed insights in Russian format.
        Generates comprehensive analysis matching sravni.ae/crypto style.
        """
        if not self.enabled:
            return {
                'confidence': 'N/A',
                'analysis': 'GPT analysis not available',
                'detailed_analysis': '',
                'recommendations': []
            }
        
        try:
            # Get detailed analysis in Russian format
            detailed_analysis = self.generate_detailed_russian_analysis(signal_data, market_data)
            
            # Get confidence and recommendations
            confidence_data = self.analyze_confidence_and_risks(signal_data, market_data)
            
            return {
                'confidence': confidence_data.get('confidence', 'Medium'),
                'analysis': confidence_data.get('summary', ''),
                'detailed_analysis': detailed_analysis,
                'recommendations': confidence_data.get('recommendations', []),
                'risk_assessment': confidence_data.get('risk_assessment', '')
            }
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return {
                'confidence': 'Error',
                'analysis': f'GPT analysis failed: {str(e)}',
                'detailed_analysis': '',
                'recommendations': []
            }
    
    def generate_detailed_russian_analysis(self, signal_data: Dict, market_data: pd.DataFrame) -> str:
        """
        Generate detailed analysis in Russian using GPT-4 (for compatibility).
        """
        if not self.enabled:
            return ""
        
        try:
            latest = market_data.iloc[-1]
            direction = 'LONG' if signal_data['direction'] == 'bullish' else 'SHORT'
            
            prompt = f"""
Создай детальный анализ криптовалютного сигнала на русском языке.

{signal_data['symbol']} – {direction}
Цена: ${signal_data['current_price']:.4f}
Entry: ${signal_data.get('entry_price', signal_data['current_price']):.4f}
Stop Loss: ${signal_data['stop_loss']:.4f}
Take Profit: ${signal_data['take_profit']:.4f}

Линии Alligator:
- Lips: ${signal_data.get('lips', latest.get('lips', 0)):.4f}
- Teeth: ${signal_data.get('teeth', latest.get('teeth', 0)):.4f}
- Jaw: ${signal_data.get('jaw', latest.get('jaw', 0)):.4f}

Создай анализ включающий:
1. Положение линий и фазу рынка
2. Оценку силы сигнала
3. Рекомендации по входу и управлению позицией
4. Ключевые риски

Ответ на русском языке, 200-300 слов."""

            # Always use GPT-4 for detailed Russian analysis to ensure compatibility
            response = self.client.chat.completions.create(
                model=self.detail_model,
                messages=[
                    {"role": "system", "content": "Ты опытный криптотрейдер. Даёшь детальный технический анализ на русском языке."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            
            if result:
                # Format nicely
                formatted = f"""
{signal_data['symbol']} – 4H, Williams Alligator

**Время анализа**: {datetime.now().strftime("%d %B %Y, %H:%M")}

{result}

**Параметры сделки**:
• Entry: ${signal_data.get('entry_price', signal_data['current_price']):.4f}
• Stop Loss: ${signal_data['stop_loss']:.4f}
• Take Profit: ${signal_data['take_profit']:.4f}
• Risk/Reward: {signal_data['risk_reward']:.2f}
"""
                logger.info(f"GPT-4 detailed analysis generated: {len(result)} chars")
                return formatted
            else:
                logger.warning("GPT-4 returned empty")
                return self._generate_fallback_analysis(signal_data, market_data)
                
        except Exception as e:
            logger.error(f"GPT-4 detailed analysis failed: {e}")
            return self._generate_fallback_analysis(signal_data, market_data)
    
    def _generate_fallback_analysis(self, signal_data: Dict, market_data: pd.DataFrame) -> str:
        """Generate fallback analysis when GPT analysis fails."""
        direction = 'LONG' if signal_data['direction'] == 'bullish' else 'SHORT'
        return f"""
{signal_data['symbol']} – 4H, Williams Alligator

**Сигнал**: {direction}
**Текущая цена**: ${signal_data['current_price']:.4f}

**Технический анализ**:
- Линии Alligator выстроились в {'восходящем' if direction == 'LONG' else 'нисходящем'} порядке
- Пробой фрактала подтвержден
- Risk/Reward: {signal_data['risk_reward']:.2f}

**Рекомендации**:
- Entry: ${signal_data.get('entry_price', signal_data['current_price']):.4f}
- Stop Loss: ${signal_data['stop_loss']:.4f}
- Take Profit: ${signal_data['take_profit']:.4f}

⚠️ Управляйте рисками. Не более 1-2% от депозита на сделку.
"""
    
    def analyze_confidence_and_risks(self, signal_data: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Analyze signal confidence, risks, and generate recommendations.
        Uses GPT-5 (gpt-o1-preview) with proper parameters if available.
        """
        if not self.enabled:
            return {'confidence': 'N/A', 'recommendations': []}
        
        try:
            context = self._prepare_context(signal_data, market_data)
            
            prompt = f"""
Analyze this crypto signal and provide confidence assessment:

Symbol: {signal_data['symbol']}
Direction: {'LONG' if signal_data['direction'] == 'bullish' else 'SHORT'}
Risk/Reward: {signal_data['risk_reward']:.2f}

{context}

Provide:
1. Confidence level: High/Medium/Low (choose one)
2. Brief summary (2-3 sentences)
3. Top 3 specific recommendations
4. Main risk factors

Format response as JSON with keys: confidence, summary, recommendations (list), risk_assessment"""

            # Use appropriate parameters based on the model
            if self.model == 'gpt-o1-preview':
                # GPT-5 (gpt-o1-preview) parameters
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=2000  # GPT-5 uses max_completion_tokens, not max_tokens
                    # Note: No temperature parameter for GPT-5 as it's not supported
                )
            else:
                # GPT-4 and other models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a crypto trading expert. Respond in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.7
                )
            
            try:
                # Try to parse as JSON
                import json
                result = json.loads(response.choices[0].message.content)
                logger.info(f"Successfully parsed JSON response from {self.model}")
                return result
            except:
                # Fallback to text parsing
                text = response.choices[0].message.content
                confidence = 'Medium'
                if 'High' in text:
                    confidence = 'High'
                elif 'Low' in text:
                    confidence = 'Low'
                
                logger.info(f"Used text parsing fallback for {self.model} response")
                return {
                    'confidence': confidence,
                    'summary': text[:200],
                    'recommendations': self._extract_recommendations(text),
                    'risk_assessment': ''
                }
                
        except Exception as e:
            logger.error(f"Confidence analysis failed with {self.model}: {e}")
            return {'confidence': 'Medium', 'recommendations': []}
    
    def _prepare_context(self, signal_data: Dict, market_data: pd.DataFrame) -> str:
        """
        Prepare market context for GPT analysis.
        """
        latest = market_data.iloc[-1]
        prev_close = market_data.iloc[-2]['close']
        
        # Calculate recent metrics
        price_change = ((latest['close'] - prev_close) / prev_close) * 100
        volatility = market_data['close'].pct_change().std() * 100
        volume_trend = 'increasing' if latest['volume'] > market_data['volume'].mean() else 'decreasing'
        
        context = f"""
        - Current Price: ${latest['close']:.4f}
        - 24h Change: {price_change:.2f}%
        - Volatility: {volatility:.2f}%
        - Volume Trend: {volume_trend}
        - Alligator Status: Челюсть=${latest['jaw']:.4f}, Зубы=${latest['teeth']:.4f}, Губы=${latest['lips']:.4f}
        """
        
        return context
    
    def _prepare_detailed_context(self, signal_data: Dict, market_data: pd.DataFrame) -> str:
        """
        Prepare detailed context for Russian GPT analysis.
        """
        latest = market_data.iloc[-1]
        
        # Calculate metrics for last 24 hours (6 candles on 4H)
        recent_data = market_data.iloc[-6:]
        price_change_24h = ((latest['close'] - recent_data.iloc[0]['close']) / recent_data.iloc[0]['close']) * 100
        
        # Volume analysis
        avg_volume = market_data['volume'].mean()
        volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1
        max_volume = market_data['volume'].max()
        
        # Alligator dynamics
        lips_teeth_dist = abs(latest['lips'] - latest['teeth'])
        teeth_jaw_dist = abs(latest['teeth'] - latest['jaw'])
        total_spread = (lips_teeth_dist + teeth_jaw_dist) / latest['close'] * 100
        
        # Trend strength
        consecutive_green = 0
        consecutive_red = 0
        for i in range(len(market_data)-1, max(0, len(market_data)-10), -1):
            if market_data.iloc[i]['close'] > market_data.iloc[i]['open']:
                consecutive_green += 1
                if consecutive_red > 0:
                    break
            else:
                consecutive_red += 1
                if consecutive_green > 0:
                    break
        
        context = f"""
Дополнительный контекст:
- Изменение за 24ч: {price_change_24h:.2f}%
- Текущий объём к среднему: {volume_ratio:.2f}x
- Максимальный объём за период: ${max_volume * latest['close']:,.0f}
- Расхождение линий Alligator: {total_spread:.2f}%
- Последовательных {'зелёных' if consecutive_green > consecutive_red else 'красных'} свечей: {max(consecutive_green, consecutive_red)}
- Волатильность (стандартное отклонение): {market_data['close'].pct_change().std() * 100:.2f}%
"""
        return context
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """
        Extract key recommendations from GPT analysis.
        """
        recommendations = []
        
        # Simple extraction based on common patterns
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'consider', 'should']):
                if line.startswith('-') or line.startswith('•'):
                    recommendations.append(line[1:].strip())
                elif len(line) < 200:  # Reasonable length for a recommendation
                    recommendations.append(line)
        
        return recommendations[:5]  # Limit to top 5 recommendations


class AlligatorFractalStrategy:
    def __init__(self, exchange='binance', timeframe='4h', use_gpt: bool = False, gpt_api_key: Optional[str] = None, gpt_model: Optional[str] = None):
        self.exchange_name = exchange
        self.timeframe = timeframe
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1200,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        self.alligator_params = {
            'jaw_period': 13,
            'jaw_shift': 8,
            'teeth_period': 8,
            'teeth_shift': 5,
            'lips_period': 5,
            'lips_shift': 3
        }
        
        self.signals = []
        self.errors = []
        
        # Initialize GPT analyzer if requested
        self.use_gpt = use_gpt
        self.gpt_analyzer = GPTAnalyzer(api_key=gpt_api_key, model=gpt_model) if use_gpt else None
        
    def fetch_top_cryptos(self, limit=30) -> List[str]:
        try:
            markets = self.exchange.fetch_tickers()
            usdt_pairs = {
                symbol: ticker for symbol, ticker in markets.items() 
                if symbol.endswith('/USDT') and ticker['quoteVolume'] is not None
            }
            
            sorted_pairs = sorted(
                usdt_pairs.items(), 
                key=lambda x: x[1]['quoteVolume'] if x[1]['quoteVolume'] else 0, 
                reverse=True
            )
            
            top_symbols = [symbol for symbol, _ in sorted_pairs[:limit]]
            print(f"Fetched top {len(top_symbols)} crypto pairs by volume")
            return top_symbols
            
        except Exception as e:
            print(f"Error fetching top cryptos: {e}")
            return self.get_default_symbols()[:limit]
    
    def get_default_symbols(self) -> List[str]:
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'TRX/USDT', 'DOT/USDT',
            'MATIC/USDT', 'LINK/USDT', 'TON/USDT', 'ICP/USDT', 'SHIB/USDT',
            'DAI/USDT', 'LTC/USDT', 'BCH/USDT', 'UNI/USDT', 'ATOM/USDT',
            'ETC/USDT', 'XLM/USDT', 'NEAR/USDT', 'FIL/USDT', 'APT/USDT',
            'ARB/USDT', 'VET/USDT', 'AAVE/USDT', 'ALGO/USDT', 'FTM/USDT'
        ]
    
    def fetch_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            
            if len(ohlcv) < 50:  # Need at least 50 candles for proper analysis
                self.errors.append(f"Insufficient data for {symbol}: only {len(ohlcv)} candles available")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.errors.append(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_smma(self, data: pd.Series, period: int) -> pd.Series:
        smma = pd.Series(index=data.index, dtype=float)
        sma = data.rolling(window=period).mean()
        smma.iloc[period-1] = sma.iloc[period-1]
        
        for i in range(period, len(data)):
            if pd.notna(smma.iloc[i-1]):
                smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + data.iloc[i]) / period
        
        return smma
    
    def calculate_alligator(self, df: pd.DataFrame) -> pd.DataFrame:
        median_price = (df['high'] + df['low']) / 2
        
        jaw_smma = self.calculate_smma(median_price, self.alligator_params['jaw_period'])
        teeth_smma = self.calculate_smma(median_price, self.alligator_params['teeth_period'])
        lips_smma = self.calculate_smma(median_price, self.alligator_params['lips_period'])
        
        df['jaw'] = jaw_smma.shift(self.alligator_params['jaw_shift'])
        df['teeth'] = teeth_smma.shift(self.alligator_params['teeth_shift'])
        df['lips'] = lips_smma.shift(self.alligator_params['lips_shift'])
        
        return df
    
    def detect_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['up_fractal'] = False
        df['down_fractal'] = False
        
        if len(df) < 5:  # Need at least 5 candles for fractal detection
            return df
        
        for i in range(2, len(df) - 2):
            high_curr = df['high'].iloc[i]
            high_prev2 = df['high'].iloc[i-2]
            high_prev1 = df['high'].iloc[i-1]
            high_next1 = df['high'].iloc[i+1]
            high_next2 = df['high'].iloc[i+2]
            
            if (high_curr > high_prev2 and high_curr > high_prev1 and 
                high_curr > high_next1 and high_curr > high_next2):
                df.iloc[i, df.columns.get_loc('up_fractal')] = True
            
            low_curr = df['low'].iloc[i]
            low_prev2 = df['low'].iloc[i-2]
            low_prev1 = df['low'].iloc[i-1]
            low_next1 = df['low'].iloc[i+1]
            low_next2 = df['low'].iloc[i+2]
            
            if (low_curr < low_prev2 and low_curr < low_prev1 and 
                low_curr < low_next1 and low_curr < low_next2):
                df.iloc[i, df.columns.get_loc('down_fractal')] = True
        
        return df
    
    def analyze_candle(self, df: pd.DataFrame, idx: int) -> Dict:
        if idx >= len(df) or idx < 0:
            return None
            
        candle = df.iloc[idx]
        prev_candle = df.iloc[idx-1] if idx > 0 else None
        
        body = abs(candle['close'] - candle['open'])
        range_total = candle['high'] - candle['low']
        
        is_green = candle['close'] > candle['open']
        is_red = candle['close'] < candle['open']
        
        volume_increase = False
        if prev_candle is not None:
            volume_increase = candle['volume'] > prev_candle['volume']
        
        return {
            'is_green': is_green,
            'is_red': is_red,
            'body': body,
            'range': range_total,
            'volume_increase': volume_increase,
            'volume': candle['volume']
        }
    
    def detect_alligator_crossing(self, df: pd.DataFrame, direction: str = 'bullish', lookback: int = 10) -> bool:
        """
        Detect if Alligator lines have recently crossed in the specified direction.
        Checks for actual crossings within the lookback period, not just current positions.
        """
        if len(df) < lookback + 1:
            return False
            
        # Get recent data
        recent_data = df.iloc[-lookback:]
        
        if direction == 'bullish':
            # For bullish: lips should cross above teeth, teeth above jaw
            # Check if lines were previously intertwined or bearish and now bullish
            for i in range(1, len(recent_data)):
                prev = recent_data.iloc[i-1]
                curr = recent_data.iloc[i]
                
                # Check for lips (губы) crossing above teeth (зубы)
                lips_crossed_teeth = (prev['lips'] <= prev['teeth'] and 
                                     curr['lips'] > curr['teeth'])
                
                # Check for teeth (зубы) crossing above jaw (челюсть)
                teeth_crossed_jaw = (prev['teeth'] <= prev['jaw'] and 
                                    curr['teeth'] > curr['jaw'])
                
                # Check if currently in bullish alignment
                currently_bullish = (curr['lips'] > curr['teeth'] > curr['jaw'])
                
                if (lips_crossed_teeth or teeth_crossed_jaw) and currently_bullish:
                    return True
                    
        else:  # bearish
            # For bearish: lips should cross below teeth, teeth below jaw
            for i in range(1, len(recent_data)):
                prev = recent_data.iloc[i-1]
                curr = recent_data.iloc[i]
                
                # Check for lips (губы) crossing below teeth (зубы)
                lips_crossed_teeth = (prev['lips'] >= prev['teeth'] and 
                                     curr['lips'] < curr['teeth'])
                
                # Check for teeth (зубы) crossing below jaw (челюсть)
                teeth_crossed_jaw = (prev['teeth'] >= prev['jaw'] and 
                                    curr['teeth'] < curr['jaw'])
                
                # Check if currently in bearish alignment
                currently_bearish = (curr['lips'] < curr['teeth'] < curr['jaw'])
                
                if (lips_crossed_teeth or teeth_crossed_jaw) and currently_bearish:
                    return True
        
        return False
    
    def check_alligator_alignment(self, df: pd.DataFrame, direction: str = 'bullish') -> bool:
        """
        Check if Alligator lines are properly aligned (not just crossed).
        This is the old check_alligator_crossing logic for backward compatibility.
        """
        latest = df.iloc[-1]
        
        if direction == 'bullish':
            return (latest['lips'] > latest['teeth'] and 
                    latest['teeth'] > latest['jaw'])
        else:
            return (latest['lips'] < latest['teeth'] and 
                    latest['teeth'] < latest['jaw'])
    
    def find_last_fractal(self, df: pd.DataFrame, fractal_type: str = 'up') -> Optional[int]:
        if fractal_type == 'up':
            fractal_mask = df['up_fractal']
        else:
            fractal_mask = df['down_fractal']
        
        for i in range(len(df) - 1, -1, -1):
            if fractal_mask.iloc[i]:
                return i
        return None
    
    def check_fractal_separation(self, df: pd.DataFrame, fractal_idx: int, direction: str = 'bullish') -> bool:
        """
        Check if ALL 5 candles of the fractal pattern DO NOT TOUCH Alligator lines.
        Requirement: ни одна из пяти свечей фрактала не должна касаться линии аллигатора
        """
        # Ensure we have enough candles for the fractal pattern
        if fractal_idx < 2 or fractal_idx >= len(df) - 2:
            return False
        
        # Get all 5 candles that form the fractal pattern
        fractal_candles = df.iloc[fractal_idx-2:fractal_idx+3]
        
        # Check that NO candle touches ANY Alligator line
        for i in range(len(fractal_candles)):
            candle = fractal_candles.iloc[i]
            candle_high = candle['high']
            candle_low = candle['low']
            
            # Get Alligator values
            jaw = candle['jaw']
            teeth = candle['teeth']
            lips = candle['lips']
            
            # Check if candle touches any Alligator line
            # A candle touches a line if the line value is between its low and high
            if (candle_low <= jaw <= candle_high or
                candle_low <= teeth <= candle_high or
                candle_low <= lips <= candle_high):
                return False
        
        return True  # If no touches, its valid
    
    def check_breakout(self, df: pd.DataFrame, fractal_idx: int, direction: str = 'bullish') -> Optional[int]:
        """
        Check for a valid breakout per requirements:
        - Пробой фрактала зеленой свечой (повышенный диапазон и повышенный объем) для long
        - Пробой фрактала красной свечой (повышенный диапазон и повышенный объем) для short
        """
        fractal_candle = df.iloc[fractal_idx]
        
        for i in range(fractal_idx + 1, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1] if i > 0 else None
            
            if direction == 'bullish':
                # Check if price breaks above fractal high
                if current_candle['close'] > fractal_candle['high'] or current_candle['high'] > fractal_candle['high']:
                    # Check if it's a green candle with increased volume and range
                    is_green = current_candle['close'] > current_candle['open']
                    
                    if prev_candle is not None:
                        volume_increase = current_candle['volume'] > prev_candle['volume']
                        range_increase = (current_candle['high'] - current_candle['low']) > (prev_candle['high'] - prev_candle['low'])
                        
                        if is_green and volume_increase and range_increase:
                            return i
            else:
                # Check if price breaks below fractal low
                if current_candle['close'] < fractal_candle['low'] or current_candle['low'] < fractal_candle['low']:
                    # Check if it's a red candle with increased volume and range
                    is_red = current_candle['close'] < current_candle['open']
                    
                    if prev_candle is not None:
                        volume_increase = current_candle['volume'] > prev_candle['volume']
                        range_increase = (current_candle['high'] - current_candle['low']) > (prev_candle['high'] - prev_candle['low'])
                        
                        if is_red and volume_increase and range_increase:
                            return i
        
        return None
    
    def check_no_touch(self, df: pd.DataFrame, fractal_idx: int, breakout_idx: int, direction: str = 'bullish') -> bool:
        """
        Check that NO candle between fractal and breakout touches Alligator lines.
        Requirement: Между зеленой свечой, пробившей фрактал и самим фракталом 
        ни одна свеча не должна касаться линий аллигатора.
        """
        for i in range(fractal_idx + 1, breakout_idx):
            candle = df.iloc[i]
            candle_high = candle['high']
            candle_low = candle['low']
            
            # Get Alligator values
            jaw = candle['jaw']
            teeth = candle['teeth']
            lips = candle['lips']
            
            # Check if candle touches any Alligator line
            # A candle touches a line if the line value is between its low and high
            if (candle_low <= jaw <= candle_high or
                candle_low <= teeth <= candle_high or
                candle_low <= lips <= candle_high):
                return False
        
        return True
    
    def calculate_sl_tp(self, df: pd.DataFrame, breakout_idx: int, direction: str = 'bullish') -> Dict:
        """
        Calculate stop loss and take profit levels per requirements:
        SL: на уровне цены синей линии челюсти аллигатора в моменте пробития фрактала
        TP: При первом закрытии свечи ниже (при long) / выше (при short) зеленой линии губ аллигатора
        """
        breakout_candle = df.iloc[breakout_idx]
        entry_price = breakout_candle['close']
        
        if direction == 'bullish':
            # SL at the blue jaw line at the moment of fractal breakout
            sl = breakout_candle['jaw']
            
            # TP: First candle CLOSE below the green lips line
            tp = None
            for i in range(breakout_idx + 1, len(df)):
                current_candle = df.iloc[i]
                
                # Check if close is below lips
                if current_candle['close'] < current_candle['lips']:
                    tp = current_candle['lips']  # Exit at lips level
                    break
            
            # If no crossing found in available data, project TP
            if tp is None:
                # Use a projection based on typical Alligator behavior
                risk = entry_price - sl
                tp = entry_price + (risk * 3)  # 3:1 as Alligator trends can be strong
            
        else:  # bearish
            # SL at the blue jaw line at the moment of fractal breakout
            sl = breakout_candle['jaw']
            
            # TP: First candle CLOSE above the green lips line
            tp = None
            for i in range(breakout_idx + 1, len(df)):
                current_candle = df.iloc[i]
                
                # Check if close is above lips
                if current_candle['close'] > current_candle['lips']:
                    tp = current_candle['lips']  # Exit at lips level
                    break
            
            # If no crossing found in available data, project TP
            if tp is None:
                # Use a projection based on typical Alligator behavior
                risk = sl - entry_price
                tp = entry_price - (risk * 3)  # 3:1 as Alligator trends can be strong
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'entry': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'risk_reward': risk_reward
        }
    
    def analyze_symbol(self, symbol: str, debug: bool = False) -> List[Dict]:
        """
        Analyze a single symbol for trading opportunities.
        """
        symbol_signals = []
        
        try:
            # Fetch market data
            df = self.fetch_ohlcv(symbol)
            if df.empty:
                return symbol_signals
            
            # Calculate indicators
            df = self.calculate_alligator(df)
            df = self.detect_fractals(df)
            
            # Check for valid signals for both directions
            for direction in ['bullish', 'bearish']:
                
                # Step 1: Check Alligator crossing
                alligator_crossed = self.detect_alligator_crossing(df, direction)
                if not alligator_crossed:
                    if debug:
                        print(f"  {symbol} {direction}: No Alligator crossing detected")
                    continue
                
                # Step 2: Check current alignment
                alignment_ok = self.check_alligator_alignment(df, direction)
                if not alignment_ok:
                    if debug:
                        print(f"  {symbol} {direction}: Alligator lines not properly aligned")
                    continue
                
                # Step 3: Find the last fractal
                fractal_type = 'down' if direction == 'bullish' else 'up'
                fractal_idx = self.find_last_fractal(df, fractal_type)
                if fractal_idx is None:
                    if debug:
                        print(f"  {symbol} {direction}: No {fractal_type} fractal found")
                    continue
                
                # Step 4: Check fractal separation from Alligator
                fractal_separated = self.check_fractal_separation(df, fractal_idx, direction)
                if not fractal_separated:
                    if debug:
                        print(f"  {symbol} {direction}: Fractal not separated from Alligator")
                    continue
                
                # Step 5: Check for valid breakout
                breakout_idx = self.check_breakout(df, fractal_idx, direction)
                if breakout_idx is None:
                    if debug:
                        print(f"  {symbol} {direction}: No valid breakout found")
                    continue
                
                # Step 6: Check no-touch rule
                no_touch = self.check_no_touch(df, fractal_idx, breakout_idx, direction)
                if not no_touch:
                    if debug:
                        print(f"  {symbol} {direction}: Candle touched Alligator lines between fractal and breakout")
                    continue
                
                # Step 7: Calculate entry, SL, TP
                trade_levels = self.calculate_sl_tp(df, breakout_idx, direction)
                
                # Validate trade parameters
                if trade_levels['risk_reward'] < 0.5:  # Minimum R:R ratio
                    if debug:
                        print(f"  {symbol} {direction}: Risk-reward ratio too low: {trade_levels['risk_reward']:.2f}")
                    continue
                
                # Create signal
                latest_candle = df.iloc[-1]
                current_price = latest_candle['close']
                
                signal = {
                    'symbol': symbol,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                    'current_price': current_price,
                    'entry_price': trade_levels['entry'],
                    'stop_loss': trade_levels['stop_loss'],
                    'take_profit': trade_levels['take_profit'],
                    'risk_reward': trade_levels['risk_reward'],
                    'fractal_idx': fractal_idx,
                    'breakout_idx': breakout_idx,
                    'lips': latest_candle['lips'],
                    'teeth': latest_candle['teeth'],
                    'jaw': latest_candle['jaw'],
                    'volume': latest_candle['volume'],
                    'validation_steps': {
                        'alligator_crossed': alligator_crossed,
                        'alignment_ok': alignment_ok,
                        'fractal_separated': fractal_separated,
                        'valid_breakout': breakout_idx is not None,
                        'no_touch': no_touch,
                        'acceptable_rr': trade_levels['risk_reward'] >= 0.5
                    }
                }
                
                # Add GPT analysis if enabled
                if self.use_gpt and self.gpt_analyzer:
                    try:
                        gpt_analysis = self.gpt_analyzer.analyze_signal(signal, df)
                        signal.update(gpt_analysis)
                        if debug:
                            print(f"  {symbol} {direction}: GPT analysis added")
                    except Exception as e:
                        if debug:
                            print(f"  {symbol} {direction}: GPT analysis failed: {e}")
                        signal.update({
                            'confidence': 'GPT Error',
                            'analysis': f'GPT analysis failed: {str(e)}',
                            'detailed_analysis': '',
                            'recommendations': []
                        })
                
                symbol_signals.append(signal)
                
                if debug:
                    print(f"  ✅ {symbol} {direction}: Valid signal found (R:R = {trade_levels['risk_reward']:.2f})")
                
        except Exception as e:
            error_msg = f"Error analyzing {symbol}: {str(e)}"
            self.errors.append(error_msg)
            if debug:
                print(f"  ❌ {error_msg}")
        
        return symbol_signals
    
    def run_analysis(self, symbols: Optional[List[str]] = None, limit: int = 30, debug: bool = False) -> Dict:
        """
        Run comprehensive analysis on specified symbols or top cryptos.
        """
        if debug:
            print(f"🚀 Starting Alligator Fractal Analysis with GPT {'enabled' if self.use_gpt else 'disabled'}")
            if self.gpt_analyzer:
                print(f"   Using model: {self.gpt_analyzer.model}")
        
        # Get symbols to analyze
        if symbols is None:
            symbols = self.fetch_top_cryptos(limit)
        
        # Reset signals and errors
        self.signals = []
        self.errors = []
        
        # Analyze each symbol
        total_symbols = len(symbols)
        for i, symbol in enumerate(symbols):
            if debug:
                print(f"📊 Analyzing {symbol} ({i+1}/{total_symbols})")
            
            symbol_signals = self.analyze_symbol(symbol, debug)
            self.signals.extend(symbol_signals)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Sort signals by risk-reward ratio
        self.signals.sort(key=lambda x: x['risk_reward'], reverse=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols_analyzed': total_symbols,
            'total_signals_found': len(self.signals),
            'signals': self.signals,
            'errors': self.errors,
            'gpt_enabled': self.use_gpt,
            'model_used': self.gpt_analyzer.model if self.gpt_analyzer else None
        }
        
        if debug:
            print(f"🎯 Analysis complete: {len(self.signals)} signals from {total_symbols} symbols")
            if self.errors:
                print(f"⚠️  {len(self.errors)} errors occurred")
        
        return results
    
    def save_signals_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save signals to JSON file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alligator_signals_{timestamp}.json"
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_signals': len(self.signals),
            'signals': self.signals,
            'errors': self.errors,
            'gpt_enabled': self.use_gpt,
            'model_used': self.gpt_analyzer.model if self.gpt_analyzer else None
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def save_signals_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save signals to CSV file.
        """
        if not self.signals:
            raise ValueError("No signals to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alligator_signals_{timestamp}.csv"
        
        # Flatten signal data for CSV
        csv_data = []
        for signal in self.signals:
            row = {
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'timestamp': signal['timestamp'],
                'current_price': signal['current_price'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'risk_reward': signal['risk_reward'],
                'lips': signal['lips'],
                'teeth': signal['teeth'],
                'jaw': signal['jaw'],
                'volume': signal['volume'],
                'confidence': signal.get('confidence', 'N/A'),
                'analysis': signal.get('analysis', ''),
                'gpt_enabled': self.use_gpt
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        return filename


def main():
    """
    Example usage of the AlligatorFractalStrategy
    """
    # Configuration
    USE_GPT = os.getenv('USE_GPT', 'false').lower() == 'true'
    GPT_API_KEY = os.getenv('OPENAI_API_KEY')
    GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-o1-preview')  # Default to GPT-5
    
    print(f"Initializing strategy with GPT: {USE_GPT}")
    if USE_GPT:
        print(f"GPT Model: {GPT_MODEL}")
    
    # Initialize strategy
    strategy = AlligatorFractalStrategy(
        exchange='binance',
        timeframe='4h',
        use_gpt=USE_GPT,
        gpt_api_key=GPT_API_KEY,
        gpt_model=GPT_MODEL
    )
    
    # Run analysis
    try:
        results = strategy.run_analysis(limit=30, debug=True)
        
        # Save results
        json_file = strategy.save_signals_to_file()
        print(f"Signals saved to: {json_file}")
        
        if results['total_signals_found'] > 0:
            csv_file = strategy.save_signals_to_csv()
            print(f"CSV saved to: {csv_file}")
            
            # Display summary
            print(f"\n📈 SUMMARY:")
            print(f"   Symbols analyzed: {results['total_symbols_analyzed']}")
            print(f"   Signals found: {results['total_signals_found']}")
            print(f"   GPT enabled: {results['gpt_enabled']}")
            if results['gpt_enabled']:
                print(f"   Model used: {results['model_used']}")
            
            if results['errors']:
                print(f"   Errors: {len(results['errors'])}")
            
            # Show top 3 signals
            print(f"\n🎯 TOP SIGNALS:")
            for i, signal in enumerate(results['signals'][:3]):
                print(f"   {i+1}. {signal['symbol']} {signal['direction'].upper()} - R:R {signal['risk_reward']:.2f}")
                if USE_GPT:
                    print(f"      Confidence: {signal.get('confidence', 'N/A')}")
        else:
            print("No signals found matching the criteria")
            
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()