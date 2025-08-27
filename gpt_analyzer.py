#!/usr/bin/env python3
"""
GPT-4 Signal Analyzer
Generates detailed trading signal analysis using OpenAI GPT-4
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
from openai import AsyncOpenAI
import logging

from signal_generator import TradingSignal

logger = logging.getLogger(__name__)

class GPTSignalAnalyzer:
    """Analyzes trading signals using GPT-4 to generate detailed reports"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def analyze_signal(self, signal: TradingSignal) -> str:
        """
        Generate detailed analysis for a trading signal
        Returns analysis in the exact format from your example
        """
        
        # Format current time for the analysis
        analysis_time = datetime.now().strftime("%d %B %Y, %H:%M CEST")
        
        # Build the analysis prompt
        prompt = f"""
Создай максимально детальный анализ торгового сигнала в точном формате как в примере ниже.

ДАННЫЕ СИГНАЛА:
{signal.to_analysis_context()}

Сила сигнала: {signal.strength}/10
Тип сигнала: {signal.signal_type}
Рекомендуемые уровни:
- Entry: {signal.entry_price} USDT
- Stop Loss: {signal.stop_loss} USDT
- Take Profit 1: {signal.take_profit_1} USDT
- Take Profit 2: {signal.take_profit_2} USDT

ФОРМАТ ОТВЕТА (строго следуй этому формату):

{signal.symbol} – {signal.timeframe}, {signal.indicator}

Время анализа: {analysis_time}

1. Положение линий
    • Опиши точное положение Lips относительно Teeth, Teeth относительно Jaw
    • Укажи точное расстояние между линиями в USDT и процентах
    • Опиши что это означает для тренда (восходящий/нисходящий/боковой)
    • Укажи фазу Аллигатора (сон/пробуждение/охота/насыщение)

2. Фаза рынка
    • Определи текущую фазу: накопление/распределение/тренд/консолидация
    • Укажи сколько времени прошло с начала фазы
    • Опиши признаки текущей фазы
    • Дай прогноз продолжительности

3. Цена относительно Alligator
    • Укажи точное положение цены относительно каждой линии
    • Рассчитай процентное расстояние от каждой линии
    • Опиши силу движения и моментум
    • Определи доминирование покупателей/продавцов

4. Фракталы
    • Опиши последний верхний фрактал и его статус (пробит/активен)
    • Опиши последний нижний фрактал и его роль
    • Укажи формирующиеся фракталы
    • Объясни значимость для стоп-лосса и целей

5. Объёмы
    • Проанализируй текущий объём относительно среднего
    • Опиши всплески объёма и их значение
    • Определи подтверждение сигнала объёмами
    • Укажи распределение покупок/продаж

6. Сигнал
    • Детально опиши сформировавшийся сигнал
    • Укажи все подтверждающие факторы
    • Определи силу и надёжность сигнала
    • Укажи условия отмены сигнала

7. Рекомендации
    • Entry: {signal.entry_price} USDT (обоснуй выбор уровня)
    • SL: {signal.stop_loss} USDT (объясни размещение)
    • TP1: {signal.take_profit_1} USDT (укажи вероятность достижения)
    • TP2: {signal.take_profit_2} USDT (для частичной фиксации)
    • Рекомендуемый размер позиции: укажи % от депозита
    • Risk/Reward: рассчитай соотношение

Используй технические термины, будь точен в цифрах, давай конкретные рекомендации.
"""

        try:
            # Call GPT-4 API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Ты опытный криптотрейдер и технический аналитик с 10-летним опытом. Ты специализируешься на методе Билла Вильямса и индикаторе Alligator. Давай только точный, детальный и практичный анализ."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            raise

    async def generate_extended_analysis(self, signal: TradingSignal) -> Dict[str, str]:
        """
        Generate multiple detailed analyses for different aspects
        Returns dictionary with different analysis sections
        """
        
        analyses = {}
        
        # Main technical analysis
        analyses['technical'] = await self.analyze_signal(signal)
        
        # Risk management analysis
        risk_prompt = f"""
Проведи детальный анализ риск-менеджмента для сигнала {signal.symbol}:

Данные позиции:
- Entry: {signal.entry_price} USDT
- Stop Loss: {signal.stop_loss} USDT
- Take Profit 1: {signal.take_profit_1} USDT
- Take Profit 2: {signal.take_profit_2} USDT
- Текущая цена: {signal.current_price} USDT

Рассчитай и опиши:
1. Размер позиции для депозитов: $1,000, $10,000, $100,000
2. Максимальный риск на сделку (в $ и %)
3. Потенциальная прибыль и R:R ratio
4. Оптимальное плечо (если применимо)
5. План масштабирования позиции
6. Trailing stop стратегию
7. Частичные фиксации прибыли
8. Хеджирование рисков
"""
        
        try:
            risk_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по риск-менеджменту в криптотрейдинге."},
                    {"role": "user", "content": risk_prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            analyses['risk_management'] = risk_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            analyses['risk_management'] = "Ошибка генерации анализа рисков"
        
        # Alternative scenarios
        scenarios_prompt = f"""
Опиши 3 возможных сценария развития для {signal.symbol} после сигнала {signal.signal_type}:

БЫЧИЙ СЦЕНАРИЙ (оптимистичный):
- Путь движения цены
- Ключевые уровни: {signal.take_profit_1}, {signal.take_profit_2}
- Признаки реализации
- Вероятность: X%

НЕЙТРАЛЬНЫЙ СЦЕНАРИЙ (консолидация):
- Диапазон движения
- Длительность
- Стратегия действий

МЕДВЕЖИЙ СЦЕНАРИЙ (стоп-лосс):
- Условия активации
- Уровень выхода: {signal.stop_loss}
- План Б

Для каждого сценария укажи конкретные действия трейдера.
"""
        
        try:
            scenarios_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты аналитик сценарного планирования в трейдинге."},
                    {"role": "user", "content": scenarios_prompt}
                ],
                temperature=0.6,
                max_tokens=1200
            )
            analyses['scenarios'] = scenarios_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Scenarios analysis error: {e}")
            analyses['scenarios'] = "Ошибка генерации сценариев"
        
        return analyses

class SignalReportGenerator:
    """Generates formatted reports from GPT analysis"""
    
    @staticmethod
    def generate_text_report(signal: TradingSignal, analysis: str) -> str:
        """Generate formatted text report"""
        
        report = f"""
{'='*80}
📊 ДЕТАЛЬНЫЙ АНАЛИЗ ТОРГОВОГО СИГНАЛА
{'='*80}

{analysis}

{'='*80}
📈 ПАРАМЕТРЫ СИГНАЛА
{'='*80}
Пара: {signal.symbol}
Таймфрейм: {signal.timeframe}
Индикатор: {signal.indicator}
Тип: {signal.signal_type}
Сила: {signal.strength}/10
Время: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
💰 ТОРГОВЫЕ УРОВНИ
{'='*80}
Entry: {signal.entry_price} USDT
Stop Loss: {signal.stop_loss} USDT
Take Profit 1: {signal.take_profit_1} USDT
Take Profit 2: {signal.take_profit_2} USDT

Risk/Reward: 1:{round((signal.take_profit_1 - signal.entry_price) / (signal.entry_price - signal.stop_loss), 2)}

{'='*80}
🤖 Сгенерировано с помощью GPT-4
{'='*80}
"""
        return report
    
    @staticmethod
    def generate_html_report(signal: TradingSignal, analyses: Dict[str, str]) -> str:
        """Generate HTML report with all analyses"""
        
        html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{signal.symbol} - Анализ сигнала {signal.signal_type}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .signal-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 30px;
            font-weight: bold;
            color: white;
            font-size: 18px;
        }}
        .signal-long {{
            background: linear-gradient(135deg, #10b981, #34d399);
        }}
        .signal-short {{
            background: linear-gradient(135deg, #ef4444, #f87171);
        }}
        .analysis-section {{
            background: rgba(255, 255, 255, 0.98);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .analysis-content {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.8;
        }}
        .levels-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .level-card {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .level-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        .level-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1f2937;
        }}
        .strength-bar {{
            width: 100%;
            height: 40px;
            background: #e5e7eb;
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .strength-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
            transition: width 0.5s ease;
        }}
        h1, h2 {{
            color: #1f2937;
            margin-bottom: 15px;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            background: #e5e7eb;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .tab:hover {{
            background: #d1d5db;
        }}
        .tab.active {{
            background: #667eea;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: white;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {signal.symbol} - Анализ торгового сигнала</h1>
            <div style="margin: 20px 0;">
                <span class="signal-badge signal-{signal.signal_type.lower()}">{signal.signal_type}</span>
                <span style="margin-left: 20px; font-size: 18px;">
                    Таймфрейм: <strong>{signal.timeframe}</strong> | 
                    Индикатор: <strong>{signal.indicator}</strong>
                </span>
            </div>
            
            <div>
                <div class="level-label">Сила сигнала</div>
                <div class="strength-bar">
                    <div class="strength-fill" style="width: {signal.strength * 10}%"></div>
                </div>
                <div style="text-align: center; font-size: 24px; font-weight: bold;">
                    {signal.strength}/10
                </div>
            </div>
            
            <div class="levels-grid">
                <div class="level-card">
                    <div class="level-label">Текущая цена</div>
                    <div class="level-value">{signal.current_price}</div>
                </div>
                <div class="level-card">
                    <div class="level-label">Entry</div>
                    <div class="level-value">{signal.entry_price}</div>
                </div>
                <div class="level-card">
                    <div class="level-label">Stop Loss</div>
                    <div class="level-value" style="color: #ef4444;">{signal.stop_loss}</div>
                </div>
                <div class="level-card">
                    <div class="level-label">Take Profit 1</div>
                    <div class="level-value" style="color: #10b981;">{signal.take_profit_1}</div>
                </div>
                <div class="level-card">
                    <div class="level-label">Take Profit 2</div>
                    <div class="level-value" style="color: #10b981;">{signal.take_profit_2}</div>
                </div>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>📈 Технический анализ</h2>
            <div class="tabs">
                <div class="tab active" onclick="showTab('technical')">Основной анализ</div>
                <div class="tab" onclick="showTab('risk')">Риск-менеджмент</div>
                <div class="tab" onclick="showTab('scenarios')">Сценарии</div>
            </div>
            
            <div id="technical" class="tab-content active">
                <div class="analysis-content">{analyses.get('technical', 'Анализ недоступен')}</div>
            </div>
            
            <div id="risk" class="tab-content">
                <div class="analysis-content">{analyses.get('risk_management', 'Анализ недоступен')}</div>
            </div>
            
            <div id="scenarios" class="tab-content">
                <div class="analysis-content">{analyses.get('scenarios', 'Анализ недоступен')}</div>
            </div>
        </div>
        
        <div class="footer">
            <p>Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Powered by GPT-4 & Custom Signal Generator</p>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""
        return html

if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def test():
        # You need to set your OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Create test signal
        from signal_generator import TradingSignal
        
        signal = TradingSignal(
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
        
        # Create analyzer
        analyzer = GPTSignalAnalyzer(api_key)
        
        print("Generating analysis...")
        analysis = await analyzer.analyze_signal(signal)
        
        # Generate report
        report = SignalReportGenerator.generate_text_report(signal, analysis)
        print(report)
    
    asyncio.run(test())