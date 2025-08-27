#!/usr/bin/env python3
"""
Signal Generator Interface
This module integrates with your existing signal generation script
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str                    # e.g., "BTCUSDT"
    timeframe: str                 # e.g., "4H"
    indicator: str                 # e.g., "Williams Alligator"
    signal_type: str              # "LONG" or "SHORT"
    strength: float               # 1-10 scale
    current_price: float          
    
    # Alligator specific data
    lips_value: Optional[float] = None
    teeth_value: Optional[float] = None
    jaw_value: Optional[float] = None
    
    # Key levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    
    # Fractal data
    upper_fractal: Optional[float] = None
    lower_fractal: Optional[float] = None
    
    # Volume data
    volume_24h: Optional[float] = None
    current_volume: Optional[float] = None
    volume_spike: Optional[bool] = False
    
    # Additional context
    market_phase: Optional[str] = None  # "awakening", "hunting", "sleeping", "satiation"
    trend_strength: Optional[str] = None  # "strong", "moderate", "weak"
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return data
    
    def to_analysis_context(self) -> str:
        """Generate context string for GPT analysis"""
        context = f"""
Текущие данные индикатора Alligator:
- Lips (зелёная, период 5): {self.lips_value} USDT
- Teeth (красная, период 8): {self.teeth_value} USDT
- Jaw (синяя, период 13): {self.jaw_value} USDT
- Текущая цена: {self.current_price} USDT

Фракталы:
- Верхний фрактал: {self.upper_fractal} USDT
- Нижний фрактал: {self.lower_fractal} USDT

Объёмы:
- 24ч объём: {self.volume_24h} USDT
- Текущий объём: {self.current_volume} USDT
- Всплеск объёма: {'Да' if self.volume_spike else 'Нет'}

Рыночная фаза: {self.market_phase}
Сила тренда: {self.trend_strength}
"""
        return context

class SignalGenerator:
    """
    Interface for your signal generation script
    Replace the generate_signal method with your actual signal generation logic
    """
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback  # Callback function to process signals
        self.active = False
        
    async def generate_signal(self) -> Optional[TradingSignal]:
        """
        THIS IS WHERE YOU INTEGRATE YOUR SIGNAL GENERATION LOGIC
        
        Replace this method with your actual signal generation code
        that finds signals on sravni.ae/crypto
        """
        
        # EXAMPLE: This is mock data - replace with your actual signal generation
        # Your script should populate these values based on real market data
        
        signal = TradingSignal(
            symbol="BTCUSDT",
            timeframe="4H",
            indicator="Williams Alligator",
            signal_type="LONG",
            strength=8.5,
            current_price=120250,
            
            # Alligator values
            lips_value=119450,
            teeth_value=118900,
            jaw_value=118200,
            
            # Entry and exit levels
            entry_price=119000,
            stop_loss=117700,
            take_profit_1=121800,
            take_profit_2=123500,
            
            # Fractals
            upper_fractal=118300,
            lower_fractal=117650,
            
            # Volume data
            volume_24h=1200000000,
            current_volume=450000000,
            volume_spike=True,
            
            # Market context
            market_phase="hunting",
            trend_strength="strong"
        )
        
        return signal
    
    async def check_for_signals(self) -> List[TradingSignal]:
        """
        Check for new signals
        This method should be called periodically to check for new signals
        """
        signals = []
        
        try:
            # Your logic to check multiple pairs/timeframes
            # For now, just checking once
            signal = await self.generate_signal()
            if signal:
                signals.append(signal)
                logger.info(f"Generated signal: {signal.symbol} {signal.signal_type} ({signal.strength}/10)")
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def start_monitoring(self, interval: int = 300):
        """
        Start continuous signal monitoring
        
        Args:
            interval: Check interval in seconds
        """
        self.active = True
        logger.info(f"Starting signal generation every {interval} seconds...")
        
        while self.active:
            try:
                # Generate signals
                signals = await self.check_for_signals()
                
                # Process each signal with callback
                if self.callback and signals:
                    for signal in signals:
                        await self.callback(signal)
                
                # Wait before next check
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Signal generation stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(60)
    
    def stop(self):
        """Stop signal generation"""
        self.active = False
        logger.info("Signal generation stopped")

# Example of how to integrate your existing script
class YourCustomSignalGenerator(SignalGenerator):
    """
    Subclass this and override generate_signal() with your actual implementation
    """
    
    async def generate_signal(self) -> Optional[TradingSignal]:
        """
        Override this method with your actual signal finding logic
        
        Example:
        1. Check sravni.ae/crypto or your data source
        2. Analyze indicators
        3. Identify signal conditions
        4. Return TradingSignal object with all data
        """
        
        # YOUR CODE HERE
        # Example structure:
        
        # Step 1: Get market data
        # market_data = await self.fetch_market_data()
        
        # Step 2: Calculate indicators
        # alligator = self.calculate_alligator(market_data)
        # fractals = self.calculate_fractals(market_data)
        
        # Step 3: Check signal conditions
        # if self.is_long_signal(alligator, fractals):
        #     return TradingSignal(...)
        
        # For now, returning parent's mock implementation
        return await super().generate_signal()

if __name__ == "__main__":
    # Test the signal generator
    async def test():
        generator = SignalGenerator()
        signal = await generator.generate_signal()
        
        if signal:
            print(f"Generated test signal:")
            print(json.dumps(signal.to_dict(), indent=2, default=str))
            print("\nContext for GPT analysis:")
            print(signal.to_analysis_context())
    
    asyncio.run(test())