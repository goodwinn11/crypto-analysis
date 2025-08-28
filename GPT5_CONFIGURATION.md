# GPT-5 Configuration for Crypto Analysis

## Implementation Status ✅

The crypto analysis system has been configured to use GPT-5 with the new Responses API, implementing medium reasoning and high verbosity for detailed Russian analysis.

## Configuration Details

### 1. Model Settings
- **Model**: `gpt-5` (configured in .env)
- **API**: Responses API (not Chat Completions)
- **Reasoning**: `medium` effort for balanced performance
- **Verbosity**: `high` for detailed Russian analysis

### 2. API Structure

The GPT-5 Responses API uses a different structure than Chat Completions:

```python
# GPT-5 Responses API (NEW)
response = client.responses.create(
    model="gpt-5",
    input="Your prompt here",
    reasoning={"effort": "medium"},  # medium reasoning
    text={"verbosity": "high"}       # high verbosity for detail
)
result = response.output_text

# vs Old Chat Completions API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "prompt"}],
    max_tokens=2000
)
```

### 3. Key Differences from GPT-4

| Feature | GPT-4 | GPT-5 |
|---------|--------|--------|
| API Endpoint | chat.completions | responses |
| Input Format | messages array | input string |
| Reasoning Control | None | effort: minimal/low/medium/high |
| Verbosity Control | None | verbosity: low/medium/high |
| Token Parameter | max_tokens | Controlled by verbosity |
| Temperature | Supported | Not needed (reasoning handles) |

### 4. Implementation in strategy.py

The GPTAnalyzer class has been updated to:
- Use GPT-5 model name (`gpt-5`)
- Support Responses API with fallback to Chat API
- Configure medium reasoning for analysis
- Use high verbosity for Russian detailed content

### 5. Russian Prompt Format

The detailed Russian analysis uses the exact format requested:

```
Ты опытный криптотрейдер с 10-летним стажем...

ДАННЫЕ СИГНАЛА:
Символ: BTC/USDT
Таймфрейм: 4H
...

ФОРМАТ ОТВЕТА:
1. Положение линий
2. Фаза рынка
3. Цена относительно Alligator
4. Фракталы
5. Объёмы
6. Сигнал
7. Рекомендации
```

## API Access Note

GPT-5 requires special API access. The system is configured correctly and will automatically use GPT-5 when:
1. Your API key has access to the `gpt-5` model
2. The Responses API is available in your OpenAI account

Until then, the system gracefully falls back to GPT-4 with appropriate parameters.

## Testing

To test GPT-5 integration:
1. Navigate to https://sravni.ae/crypto
2. Click "Запустить анализ" button
3. Wait 1-2 minutes for analysis
4. Check for detailed Russian analysis in results

## Files Modified
- `/home/ubuntu/SravniAe/strategy.py` - GPT-5 implementation
- `/home/ubuntu/SravniAe/.env` - Model set to `gpt-5`
- Docker container synchronized with updates

## Next Steps

Once your API key has GPT-5 access:
1. The system will automatically use the Responses API
2. Analysis will use medium reasoning for balanced performance
3. Detailed Russian analysis will use high verbosity
4. No code changes needed - it's ready to go