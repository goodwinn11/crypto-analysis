#!/usr/bin/env python3
"""
Configuration management for Crypto Signal Analysis System
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """System configuration"""
    
    # OpenAI API
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    GPT_MODEL: str = os.getenv('GPT_MODEL', 'gpt-4-turbo-preview')
    GPT_TEMPERATURE: float = float(os.getenv('GPT_TEMPERATURE', '0.7'))
    GPT_MAX_TOKENS: int = int(os.getenv('GPT_MAX_TOKENS', '2000'))
    
    # Monitoring
    CHECK_INTERVAL: int = int(os.getenv('CHECK_INTERVAL', '300'))
    OUTPUT_DIR: Path = Path(os.getenv('OUTPUT_DIR', 'signal_reports'))
    
    # Signal Processing
    MAX_CONCURRENT_ANALYSES: int = int(os.getenv('MAX_CONCURRENT_ANALYSES', '3'))
    SIGNAL_STRENGTH_THRESHOLD: float = float(os.getenv('SIGNAL_STRENGTH_THRESHOLD', '7.0'))
    
    # Telegram Integration (optional)
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv('TELEGRAM_CHAT_ID')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'crypto_signals.log')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            print("❌ OPENAI_API_KEY is not set!")
            return False
        
        # Create output directory if it doesn't exist
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("\n" + "="*50)
        print("CONFIGURATION")
        print("="*50)
        print(f"OpenAI API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Not set'}")
        print(f"GPT Model: {cls.GPT_MODEL}")
        print(f"Check Interval: {cls.CHECK_INTERVAL} seconds")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"Signal Threshold: {cls.SIGNAL_STRENGTH_THRESHOLD}/10")
        print(f"Telegram Bot: {'✅ Configured' if cls.TELEGRAM_BOT_TOKEN else '❌ Not configured'}")
        print("="*50 + "\n")