#!/usr/bin/env python3
"""
Lightning Discord TTS Bot - Main Entry Point
Usage: python run.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from bot.main import main

if __name__ == "__main__":
    try:
        print("🚀 Starting Lightning Discord TTS Bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
