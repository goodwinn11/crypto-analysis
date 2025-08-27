# 🚀 Crypto Signal Analysis System with GPT-4

Система автоматического анализа криптовалютных сигналов с использованием GPT-4 для генерации детальных торговых рекомендаций.

## 📋 Описание

Система получает торговые сигналы от вашего скрипта и автоматически генерирует детальный анализ через GPT-4, включая:

- Положение линий индикатора Williams Alligator
- Анализ фазы рынка
- Уровни фракталов
- Объёмный анализ
- Детальные рекомендации по входу/выходу
- Риск-менеджмент
- Альтернативные сценарии

## 🛠 Установка

```bash
# 1. Клонируйте или скопируйте файлы

# 2. Установите зависимости
pip install -r requirements.txt

# 3. Установите OpenAI API ключ
export OPENAI_API_KEY='your-api-key-here'
```

## 🔧 Интеграция с вашим скриптом

### Вариант 1: Модификация signal_generator.py

Откройте `signal_generator.py` и замените метод `generate_signal()` вашей логикой:

```python
class YourCustomSignalGenerator(SignalGenerator):
    async def generate_signal(self) -> Optional[TradingSignal]:
        # ВАШ КОД ЗДЕСЬ
        # 1. Получите данные с sravni.ae/crypto или вашего источника
        # 2. Проанализируйте индикаторы
        # 3. Определите условия сигнала
        # 4. Верните объект TradingSignal со всеми данными
        
        if signal_detected:  # ваша логика
            return TradingSignal(
                symbol="BTCUSDT",
                timeframe="4H",
                indicator="Williams Alligator",
                signal_type="LONG",  # или "SHORT"
                strength=8.5,         # сила сигнала 1-10
                current_price=120250,
                
                # Значения Alligator
                lips_value=119450,
                teeth_value=118900,
                jaw_value=118200,
                
                # Торговые уровни
                entry_price=119000,
                stop_loss=117700,
                take_profit_1=121800,
                take_profit_2=123500,
                
                # Дополнительные данные
                upper_fractal=118300,
                lower_fractal=117650,
                volume_24h=1200000000,
                volume_spike=True,
                market_phase="hunting",
                trend_strength="strong"
            )
        return None
```

### Вариант 2: Создание отдельного модуля

Создайте файл `my_signals.py`:

```python
from signal_generator import SignalGenerator, TradingSignal
import asyncio

class MySignalGenerator(SignalGenerator):
    async def generate_signal(self):
        # Ваш код для поиска сигналов
        signal_data = await self.fetch_from_sravni()  # ваш метод
        
        if signal_data:
            return TradingSignal(**signal_data)
        return None
    
    async def fetch_from_sravni(self):
        # Ваша логика получения данных
        pass
```

## 📊 Использование

### Тестовый запуск (с примером сигнала):

```bash
python main.py --test
```

### Продакшн режим (с вашим генератором):

```bash
# Используя базовый генератор (нужно модифицировать)
python main.py

# Используя кастомный генератор
python main.py --use-custom

# С настройкой интервала (проверка каждую минуту)
python main.py --interval 60
```

### Параметры командной строки:

- `--api-key` - OpenAI API ключ (или через переменную окружения)
- `--interval` - Интервал проверки в секундах (по умолчанию: 300)
- `--output-dir` - Папка для сохранения отчетов
- `--test` - Тестовый режим с одним сигналом
- `--use-custom` - Использовать YourCustomSignalGenerator

## 📁 Структура проекта

```
crypto-gpt/
├── signal_generator.py   # Интерфейс для вашего генератора сигналов
├── gpt_analyzer.py       # GPT-4 анализатор
├── main.py              # Основной координатор
├── requirements.txt     # Python зависимости
├── config.py           # Конфигурация (опционально)
└── analysis_reports/   # Папка с отчетами
    ├── BTCUSDT_LONG_20250827_143022.txt   # Текстовый отчет
    └── BTCUSDT_LONG_20250827_143022.html  # HTML отчет
```

## 📝 Формат выходных данных

### Текстовый отчет (пример):

```
BTCUSDT – 4H, Williams Alligator

Время анализа: 27 августа 2025, 14:30 CEST

1. Положение линий
    • Lips (зелёная) выше Teeth (красная), а Teeth выше Jaw (синяя) → чёткий восходящий тренд
    • Расстояние между линиями увеличено → тренд в активной фазе

2. Фаза рынка
    • Пробуждение произошло после короткой фазы сна
    • Сейчас — середина активной фазы тренда

3. Цена относительно Alligator
    • Цена находится значительно выше всех трёх линий
    • После резкого роста

[... продолжение анализа ...]

7. Рекомендации
    • Entry: 118 800–119 200 USDT
    • SL: ниже 117 700 USDT
    • TP1: 121 800 USDT
    • TP2: 123 500 USDT
```

### HTML отчет:

- Интерактивный веб-интерфейс
- Вкладки для разных типов анализа
- Визуализация силы сигнала
- Цветовое кодирование уровней

## 🔑 Получение OpenAI API ключа

1. Зайдите на https://platform.openai.com/api-keys
2. Создайте новый API ключ
3. Сохраните его в безопасном месте
4. Установите как переменную окружения:
   ```bash
   export OPENAI_API_KEY='sk-...'
   ```

## ⚙️ Настройка

Создайте файл `.env` для хранения конфигурации:

```env
# API ключи
OPENAI_API_KEY=your-key-here

# Настройки мониторинга
CHECK_INTERVAL=300
OUTPUT_DIR=analysis_reports

# GPT настройки
GPT_MODEL=gpt-4-turbo-preview
GPT_TEMPERATURE=0.7
GPT_MAX_TOKENS=2000
```

## 📊 Мониторинг и логи

Система создает лог-файл `signal_analysis.log` со всеми событиями:

```bash
# Просмотр логов в реальном времени
tail -f signal_analysis.log

# Поиск ошибок
grep ERROR signal_analysis.log
```

## 🚨 Важные замечания

1. **API лимиты**: GPT-4 имеет ограничения на количество запросов. Рекомендуется интервал не менее 60 секунд.

2. **Стоимость**: Каждый анализ использует ~2000-4000 токенов GPT-4. Проверьте тарифы на https://openai.com/pricing

3. **Безопасность**: Никогда не публикуйте ваш API ключ. Используйте переменные окружения.

4. **Точность**: Система генерирует анализ на основе предоставленных данных. Качество анализа зависит от качества входных данных.

## 📧 Поддержка

При возникновении вопросов:

1. Проверьте логи: `cat signal_analysis.log`
2. Убедитесь, что API ключ установлен правильно
3. Проверьте формат данных в `signal_generator.py`

## 📜 Лицензия

Этот проект предназначен для образовательных целей. Используйте на свой риск при реальной торговле.