"""
Flask routes for Crypto GPT-5 Analysis
Add these routes to your Flask app.py file
"""

from flask import jsonify, request, send_from_directory
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

# Add this route to serve the crypto-gpt5 page
@app.route('/crypto-gpt5')
def crypto_gpt5_page():
    """Serve the crypto GPT-5 analysis page."""
    response = send_from_directory(app.static_folder, 'crypto-gpt5.html')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Enhanced crypto signals API endpoint
@app.route('/api/crypto/signals', methods=['GET'])
def get_crypto_signals():
    """
    Get cryptocurrency trading signals using Alligator-Fractal strategy.
    
    Query params:
        limit: Number of cryptocurrencies to analyze (default: 30)
        use_gpt: Whether to use GPT analysis (default: false)
    
    Returns:
    {
        "timestamp": "...",
        "exchange": "binance",
        "timeframe": "4h",
        "symbols_analyzed": 30,
        "signals_found": 3,
        "signals": [...]
    }
    """
    try:
        limit = int(request.args.get('limit', 30))
        limit = min(max(limit, 1), 30)  # Limit between 1 and 30
        use_gpt = request.args.get('use_gpt', 'false').lower() == 'true'
        
        logger.info(f"Fetching crypto trading signals for {limit} symbols (GPT: {use_gpt})")
        
        # Import strategy module
        from strategy import AlligatorFractalStrategy
        
        # Initialize strategy with optional GPT support
        gpt_api_key = os.getenv('OPENAI_API_KEY') if use_gpt else None
        gpt_model = os.getenv('GPT_MODEL', 'gpt-4-turbo-preview') if use_gpt else None
        
        strategy = AlligatorFractalStrategy(
            exchange='binance',
            timeframe='4h',
            use_gpt=use_gpt,
            gpt_api_key=gpt_api_key,
            gpt_model=gpt_model
        )
        
        # Run analysis
        results = strategy.run_analysis(symbols=None, limit=limit)
        
        # Ensure all values are JSON serializable
        if 'gpt_enabled' in results:
            results['gpt_enabled'] = str(results['gpt_enabled'])
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting crypto signals: {e}")
        
        # Return mock data as fallback
        mock_signals = []
        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        
        for i in range(min(2, limit)):
            signal = {
                'symbol': random.choice(pairs),
                'direction': random.choice(['LONG', 'SHORT']),
                'entry_price': round(random.uniform(30000, 60000), 2),
                'stop_loss': round(random.uniform(29000, 59000), 2),
                'take_profit': round(random.uniform(31000, 61000), 2),
                'risk_reward': round(random.uniform(1.5, 3.5), 2),
                'confidence': round(random.uniform(70, 95), 1),
                'timestamp': datetime.now().isoformat()
            }
            
            if use_gpt:
                signal['gpt_analysis'] = {
                    'market_context': 'Market analysis based on current conditions',
                    'entry_reasoning': 'Technical indicators suggest favorable entry',
                    'risk_assessment': 'Risk/reward ratio within acceptable range',
                    'confidence_score': signal['confidence']
                }
            
            mock_signals.append(signal)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'exchange': 'binance',
            'timeframe': '4h',
            'symbols_analyzed': limit,
            'signals_found': len(mock_signals),
            'signals': mock_signals,
            'gpt_enabled': str(use_gpt),
            'error': 'Using mock data due to strategy error'
        })