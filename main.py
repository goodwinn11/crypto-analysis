#!/usr/bin/env python3
"""
Main Crypto Signal Analysis System
Connects your signal generator with GPT-4 analysis
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict

# Import components
from signal_generator import SignalGenerator, TradingSignal, YourCustomSignalGenerator
from gpt_analyzer import GPTSignalAnalyzer, SignalReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalAnalysisSystem:
    """
    Main system that coordinates signal generation and GPT analysis
    """
    
    def __init__(self, openai_api_key: str, output_dir: str = "analysis_reports"):
        """
        Initialize the analysis system
        
        Args:
            openai_api_key: OpenAI API key for GPT-4
            output_dir: Directory to save analysis reports
        """
        self.analyzer = GPTSignalAnalyzer(openai_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Track processed signals
        self.processed_signals_file = self.output_dir / "processed_signals.json"
        self.processed_signals = self._load_processed_signals()
        
        # Statistics
        self.stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'analyses_generated': 0
        }
    
    def _load_processed_signals(self) -> set:
        """Load previously processed signals to avoid duplicates"""
        if self.processed_signals_file.exists():
            try:
                with open(self.processed_signals_file, 'r') as f:
                    return set(json.load(f))
            except:
                pass
        return set()
    
    def _save_processed_signals(self):
        """Save processed signal IDs"""
        with open(self.processed_signals_file, 'w') as f:
            json.dump(list(self.processed_signals), f)
    
    def _generate_signal_id(self, signal: TradingSignal) -> str:
        """Generate unique ID for a signal"""
        timestamp_str = signal.timestamp.strftime('%Y%m%d_%H%M%S')
        return f"{signal.symbol}_{signal.signal_type}_{timestamp_str}"
    
    async def process_signal(self, signal: TradingSignal):
        """
        Process a signal and generate GPT-4 analysis
        
        Args:
            signal: Trading signal to analyze
        """
        signal_id = self._generate_signal_id(signal)
        
        # Check if already processed
        if signal_id in self.processed_signals:
            logger.info(f"Signal {signal_id} already processed, skipping...")
            return
        
        try:
            # Print signal detection
            print("\n" + "="*80)
            print(f"🚨 НОВЫЙ СИГНАЛ ОБНАРУЖЕН")
            print("="*80)
            print(f"Пара: {signal.symbol}")
            print(f"Тип: {signal.signal_type}")
            print(f"Сила: {signal.strength}/10")
            print(f"Таймфрейм: {signal.timeframe}")
            print(f"Индикатор: {signal.indicator}")
            print("-"*80)
            print("Генерирую детальный анализ с помощью GPT-4...")
            
            # Generate GPT-4 analyses
            logger.info(f"Generating GPT-4 analysis for {signal.symbol} {signal.signal_type}")
            
            # Get main analysis
            main_analysis = await self.analyzer.analyze_signal(signal)
            
            # Get extended analyses (risk, scenarios)
            extended_analyses = await self.analyzer.generate_extended_analysis(signal)
            extended_analyses['technical'] = main_analysis
            
            # Save text report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            text_filename = f"{signal.symbol}_{signal.signal_type}_{timestamp}.txt"
            text_path = self.output_dir / text_filename
            
            text_report = SignalReportGenerator.generate_text_report(signal, main_analysis)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_report)
                
                # Add risk and scenarios sections
                if 'risk_management' in extended_analyses:
                    f.write("\n\n" + "="*80)
                    f.write("\n💼 РИСК-МЕНЕДЖМЕНТ\n")
                    f.write("="*80 + "\n")
                    f.write(extended_analyses['risk_management'])
                
                if 'scenarios' in extended_analyses:
                    f.write("\n\n" + "="*80)
                    f.write("\n🎯 АЛЬТЕРНАТИВНЫЕ СЦЕНАРИИ\n")
                    f.write("="*80 + "\n")
                    f.write(extended_analyses['scenarios'])
            
            logger.info(f"Text report saved to {text_path}")
            
            # Save HTML report
            html_filename = f"{signal.symbol}_{signal.signal_type}_{timestamp}.html"
            html_path = self.output_dir / html_filename
            
            html_report = SignalReportGenerator.generate_html_report(signal, extended_analyses)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"HTML report saved to {html_path}")
            
            # Print summary to console
            print("\n✅ АНАЛИЗ ЗАВЕРШЕН")
            print("-"*80)
            
            # Print first part of analysis
            lines = main_analysis.split('\n')[:20]
            print('\n'.join(lines))
            print("\n... (полный анализ в файлах)")
            
            print("-"*80)
            print(f"📄 Текстовый отчет: {text_path}")
            print(f"🌐 HTML отчет: {html_path}")
            print("="*80 + "\n")
            
            # Mark as processed
            self.processed_signals.add(signal_id)
            self._save_processed_signals()
            
            # Update stats
            self.stats['total_signals'] += 1
            self.stats['analyses_generated'] += 1
            if signal.signal_type == "LONG":
                self.stats['long_signals'] += 1
            else:
                self.stats['short_signals'] += 1
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            print(f"❌ Ошибка при обработке сигнала: {e}")
    
    async def run_continuous_monitoring(self, signal_generator: SignalGenerator, interval: int = 300):
        """
        Run continuous monitoring with your signal generator
        
        Args:
            signal_generator: Your signal generator instance
            interval: Check interval in seconds
        """
        print(f"🚀 Запуск мониторинга сигналов (проверка каждые {interval} секунд)")
        print(f"📁 Отчеты сохраняются в: {self.output_dir}")
        print("-"*80)
        
        # Set callback for signal processing
        signal_generator.callback = self.process_signal
        
        # Start monitoring
        await signal_generator.start_monitoring(interval)
    
    def print_stats(self):
        """Print system statistics"""
        print("\n" + "="*80)
        print("📊 СТАТИСТИКА СИСТЕМЫ")
        print("="*80)
        print(f"Всего сигналов: {self.stats['total_signals']}")
        print(f"LONG сигналов: {self.stats['long_signals']}")
        print(f"SHORT сигналов: {self.stats['short_signals']}")
        print(f"Анализов создано: {self.stats['analyses_generated']}")
        print("="*80)

async def test_single_signal(api_key: str):
    """Test with a single signal"""
    system = SignalAnalysisSystem(api_key)
    
    # Create test signal
    test_signal = TradingSignal(
        symbol="BTCUSDT",
        timeframe="4H",
        indicator="Williams Alligator",
        signal_type="LONG",
        strength=8.5,
        current_price=120250,
        lips_value=119450,
        teeth_value=118900,
        jaw_value=118200,
        entry_price=119000,
        stop_loss=117700,
        take_profit_1=121800,
        take_profit_2=123500,
        upper_fractal=118300,
        lower_fractal=117650,
        volume_24h=1200000000,
        current_volume=450000000,
        volume_spike=True,
        market_phase="hunting",
        trend_strength="strong"
    )
    
    # Process signal
    await system.process_signal(test_signal)
    system.print_stats()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Crypto Signal Analysis System with GPT-4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuous monitoring
  python main.py --api-key YOUR_KEY
  
  # Test with single signal
  python main.py --api-key YOUR_KEY --test
  
  # Custom interval (check every minute)
  python main.py --api-key YOUR_KEY --interval 60
  
  # Use environment variable for API key
  export OPENAI_API_KEY=your-key-here
  python main.py
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env variable)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Signal check interval in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_reports',
        help='Directory for analysis reports (default: analysis_reports)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with a single signal'
    )
    
    parser.add_argument(
        '--use-custom',
        action='store_true',
        help='Use YourCustomSignalGenerator instead of base SignalGenerator'
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n❌ OpenAI API ключ не найден!\n")
        print("Установите ключ одним из способов:")
        print("1. Переменная окружения: export OPENAI_API_KEY='your-key'")
        print("2. Аргумент командной строки: --api-key 'your-key'")
        print("\nПолучить ключ: https://platform.openai.com/api-keys")
        return
    
    # Print header
    print("\n" + "="*80)
    print("🤖 СИСТЕМА АНАЛИЗА КРИПТОВАЛЮТНЫХ СИГНАЛОВ")
    print("="*80)
    print(f"📊 Модель: GPT-4 Turbo")
    print(f"⏱  Интервал проверки: {args.interval} секунд")
    print(f"📁 Папка отчетов: {args.output_dir}")
    print("="*80 + "\n")
    
    # Create system
    system = SignalAnalysisSystem(api_key, args.output_dir)
    
    if args.test:
        # Test mode
        print("🧪 Тестовый режим - генерация одного сигнала\n")
        await test_single_signal(api_key)
    else:
        # Production mode
        # Choose signal generator
        if args.use_custom:
            print("📡 Используется: YourCustomSignalGenerator")
            print("⚠️  Убедитесь, что вы реализовали метод generate_signal()\n")
            generator = YourCustomSignalGenerator()
        else:
            print("📡 Используется: базовый SignalGenerator")
            print("ℹ️  Для использования вашего генератора, добавьте флаг --use-custom\n")
            generator = SignalGenerator()
        
        try:
            # Run continuous monitoring
            await system.run_continuous_monitoring(generator, args.interval)
        except KeyboardInterrupt:
            print("\n\n⛔ Мониторинг остановлен пользователем")
            system.print_stats()

if __name__ == "__main__":
    asyncio.run(main())