"""
Utility functions and configurations for the Lightning Discord TTS Bot
"""

import os
import json
import torch
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import time

# Load environment variables
load_dotenv()

# Constants
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# File paths
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
VOICE_CACHE_DIR = DATA_DIR / "voice_cache"
PHRASE_CACHE_DIR = DATA_DIR / "phrase_cache"
VOICE_BACKUPS_DIR = DATA_DIR / "voice_backups"

# Data files
USER_VOICE_FILE = DATA_DIR / "user_voices.json"
TTS_STATE_FILE = DATA_DIR / "tts_states.json"
BLACKLIST_FILE = DATA_DIR / "blacklist.json"
USER_SETTINGS_FILE = DATA_DIR / "user_settings.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"

# Bot settings
DEFAULT_TTS_ENGINE = "coqui"
MAX_TEXT_LENGTH = 500
MAX_REPEAT_CHARS = 5

def setup_directories():
    """Create all required directories"""
    directories = [
        DATA_DIR,
        UPLOAD_DIR,
        VOICE_CACHE_DIR,
        PHRASE_CACHE_DIR,
        VOICE_BACKUPS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def load_json_file(file_path: Path, default_value=None):
    """Load JSON file with default value if not exists"""
    if default_value is None:
        default_value = {}
    
    if not file_path.exists():
        save_json_file(file_path, default_value)
        return default_value
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return default_value

def save_json_file(file_path: Path, data):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def setup_gpu():
    """Setup GPU with maximum performance optimizations"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available - using CPU")
        return "cpu"
    
    try:
        device = "cuda:0"
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Initialized: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"⚡ Performance Mode: MAXIMUM")
        
        return device
        
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        return "cpu"

def check_admin_permissions(message):
    """Check if user has admin permissions"""
    try:
        if hasattr(message.author, 'guild_permissions'):
            return message.author.guild_permissions.administrator
        member = message.guild.get_member(message.author.id)
        if member and hasattr(member, 'guild_permissions'):
            return member.guild_permissions.administrator
        if message.author.id == message.guild.owner_id:
            return True
        return False
    except Exception:
        return False

def clean_corrupted_cache(user_id):
    """Clean up corrupted cache files for a user"""
    try:
        user_voices = load_json_file(USER_VOICE_FILE)
        cache_key = f"{user_id}_coqui_lightning"
        
        if cache_key in user_voices:
            cache_file = Path(user_voices[cache_key])
            if cache_file.exists():
                cache_file.unlink()
            del user_voices[cache_key]
            save_json_file(USER_VOICE_FILE, user_voices)
            print(f"🧹 Cleaned up corrupted cache for user {user_id}")
        
        # Clean up any orphaned cache files
        for cache_file in VOICE_CACHE_DIR.glob(f"{user_id}_*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass
                
    except Exception as e:
        print(f"❌ Error cleaning cache for user {user_id}: {e}")

class ContentFilter:
    """Content filter for auto-moderation"""
    def __init__(self):
        self.blocked_words = {'spam', 'test' * 10, 'a' * 20}
        self.max_length = MAX_TEXT_LENGTH
        self.max_repeat_chars = MAX_REPEAT_CHARS
    
    def filter_text(self, text):
        """Filter and clean text for TTS"""
        import re
        
        # Remove excessive repetition
        cleaned = re.sub(r'(.)\1{' + str(self.max_repeat_chars) + ',}', 
                        r'\1' * self.max_repeat_chars, text)
        
        # Length limit
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length] + "..."
        
        # Check for blocked content
        if any(word in cleaned.lower() for word in self.blocked_words):
            return None, "Content filtered"
        
        return cleaned, "OK"

# Initialize content filter
content_filter = ContentFilter()

# Global data stores
user_stats = defaultdict(lambda: {"phrases": [], "last_used": time.time()})
phrase_cache = {}
