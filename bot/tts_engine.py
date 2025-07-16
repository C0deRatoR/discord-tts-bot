"""
TTS Engine Module for Lightning Discord TTS Bot
Handles Coqui TTS and ElevenLabs integration with robust error handling
"""

import os
import torch
import asyncio
import requests
import time
import pickle
from pathlib import Path
from TTS.api import TTS
from elevenlabs import ElevenLabs, save
from concurrent.futures import ThreadPoolExecutor

from .utils import (
    ELEVENLABS_API_KEY, 
    setup_gpu, 
    phrase_cache,
    user_stats,
    VOICE_CACHE_DIR,
    load_json_file,
    save_json_file,
    USER_VOICE_FILE,
    clean_corrupted_cache
)

# Initialize GPU
device = setup_gpu()

# Initialize TTS engines
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=8)

# CUDA streams for parallel processing
if device.startswith("cuda"):
    cuda_streams = [torch.cuda.Stream() for _ in range(4)]
    current_stream = 0

class CoquiTTSEngine:
    """Coqui TTS Engine with GPU acceleration and robust error handling"""
    
    def __init__(self):
        self.tts = None
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize Coqui TTS with GPU optimization"""
        try:
            print("🎯 Loading Coqui TTS with Lightning Performance...")
            
            # Load model with GPU support
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", 
                          gpu=device.startswith("cuda"))
            
            if device.startswith("cuda"):
                torch.set_float32_matmul_precision('high')
                torch.set_default_device(device)
                print("⚡ GPU optimizations applied")
            
            print("✅ Coqui TTS ready with Lightning Speed")
            
        except Exception as e:
            print(f"❌ Coqui TTS initialization failed: {e}")
            self.tts = None
    
    def validate_voice_cache(self, cache_file):
        """Validate voice cache file"""
        try:
            if not Path(cache_file).exists():
                return False, "Cache file not found"
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check required fields
            required_fields = ['audio_file', 'voice_name', 'user_id']
            for field in required_fields:
                if field not in cache_data:
                    return False, f"Missing field: {field}"
            
            # Check if audio file exists
            if not Path(cache_data['audio_file']).exists():
                return False, "Audio file not found"
            
            return True, "Valid cache"
            
        except Exception as e:
            return False, f"Cache validation error: {e}"
    
    def _get_user_voice(self, user_id):
        """Get user's voice file or default with comprehensive error handling"""
        user_voices = load_json_file(Path(USER_VOICE_FILE))
        
        # Try lightning-processed voice first
        lightning_cache = user_voices.get(f"{user_id}_coqui_lightning")
        if lightning_cache and Path(lightning_cache).exists():
            is_valid, reason = self.validate_voice_cache(lightning_cache)
            if is_valid:
                try:
                    with open(lightning_cache, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    audio_file = cache_data.get('audio_file')
                    if audio_file and Path(audio_file).exists():
                        print(f"✅ Using lightning-processed voice for user {user_id}")
                        return audio_file
                        
                except Exception as e:
                    print(f"⚠️ Error loading lightning cache: {e}")
            
            # Clean up invalid cache
            print(f"⚠️ Lightning cache invalid for user {user_id}: {reason}")
            clean_corrupted_cache(user_id)
        
        # Try regular voice
        regular_voice = user_voices.get(f"{user_id}_coqui")
        if regular_voice and Path(regular_voice).exists():
            print(f"✅ Using regular voice for user {user_id}")
            return regular_voice
        
        # Default voice
        default_voice = "default_speaker.wav"
        if Path(default_voice).exists():
            print(f"✅ Using default voice for user {user_id}")
            return default_voice
        
        raise Exception("No voice available. Please upload a voice with !uploadvoice")
    
    async def generate_tts(self, text, user_id, speaker_wav=None):
        """Generate TTS with caching and GPU acceleration"""
        if not self.tts:
            raise Exception("Coqui TTS not available")
        
        try:
            # Check phrase cache first
            cache_key = f"{user_id}_{text.lower()}"
            if cache_key in phrase_cache:
                cached_file = phrase_cache[cache_key]
                if cached_file and Path(cached_file).exists():
                    print(f"⚡ Cache hit for: '{text[:30]}...'")
                    return cached_file
            
            # GPU optimization
            if device.startswith("cuda"):
                global current_stream
                stream = cuda_streams[current_stream % len(cuda_streams)]
                current_stream += 1
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            
            # Get speaker voice with error handling
            if not speaker_wav:
                speaker_wav = self._get_user_voice(user_id)
            
            output_file = f"data/phrase_cache/lightning_tts_{user_id}_{int(time.time())}.wav"
            
            def tts_generation():
                """TTS generation function"""
                start_time = time.time()
                
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language="en",
                    file_path=output_file,
                    speed=1.1
                )
                
                generation_time = time.time() - start_time
                print(f"⚡ TTS: {generation_time:.2f}s")
                
                # Cache for future use
                phrase_cache[cache_key] = output_file
                
                return output_file
            
            # Run with maximum concurrency
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, tts_generation)
            
            # Update user stats
            user_stats[user_id]["phrases"].append(text.lower())
            user_stats[user_id]["last_used"] = time.time()
            
            return result
            
        except Exception as e:
            # Clean up cache if voice-related error
            if "voice" in str(e).lower() or "audio" in str(e).lower():
                clean_corrupted_cache(user_id)
            raise Exception(f"Coqui TTS Error: {e}")

class ElevenLabsEngine:
    """ElevenLabs TTS Engine with proper initialization"""
    
    def __init__(self):
        self.client = elevenlabs_client
        self.voices = {}
        self.current_voice_id = None
        self.voices_loaded = False
    
    async def initialize(self):
        """Initialize the ElevenLabs engine - call this after event loop starts"""
        if self.client and not self.voices_loaded:
            await self.fetch_voices()
    
    async def fetch_voices(self):
        """Fetch available ElevenLabs voices"""
        if not ELEVENLABS_API_KEY:
            return
        
        try:
            headers = {"xi-api-key": ELEVENLABS_API_KEY}
            resp = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            voices_data = data.get("voices", [])
            self.voices = {v['name']: v['voice_id'] for v in voices_data}
            
            if self.voices and not self.current_voice_id:
                self.current_voice_id = list(self.voices.values())[0]
            
            self.voices_loaded = True
            print("✅ ElevenLabs voices loaded")
            
        except Exception as e:
            print(f"❌ Failed to fetch ElevenLabs voices: {e}")
    
    async def generate_tts(self, text, user_id, voice_id=None):
        """Generate TTS using ElevenLabs"""
        if not self.client:
            raise Exception("ElevenLabs not configured")
        
        try:
            # Get voice ID
            if not voice_id:
                voice_id = self._get_user_voice_id(user_id)
            
            # Generate TTS
            audio = self.client.text_to_speech.convert(text=text, voice_id=voice_id)
            
            output_file = f"data/phrase_cache/elevenlabs_tts_{user_id}_{int(time.time())}.mp3"
            save(audio, output_file)
            
            return output_file
            
        except Exception as e:
            raise Exception(f"ElevenLabs TTS Error: {e}")
    
    def _get_user_voice_id(self, user_id):
        """Get user's ElevenLabs voice ID"""
        user_voices = load_json_file(Path(USER_VOICE_FILE))
        
        voice_id = user_voices.get(f"{user_id}_elevenlabs")
        if voice_id:
            return voice_id
        
        if self.current_voice_id:
            return self.current_voice_id
        
        raise Exception("No voice selected. Use !voice <name> to select a voice.")
    
    async def upload_voice(self, user_id, audio_file, voice_name):
        """Upload voice to ElevenLabs"""
        try:
            with open(audio_file, "rb") as f:
                response = requests.post(
                    "https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": ELEVENLABS_API_KEY},
                    data={
                        'name': voice_name,
                        'description': f"Voice of user {user_id}",
                        'labels': '{"type":"user"}'
                    },
                    files={'files': (Path(audio_file).name, f, 'audio/wav')}
                )
                response.raise_for_status()
                
                voice_id = response.json()["voice_id"]
                
                # Update user voices
                user_voices = load_json_file(Path(USER_VOICE_FILE))
                user_voices[f"{user_id}_elevenlabs"] = voice_id
                save_json_file(Path(USER_VOICE_FILE), user_voices)
                
                # Refresh voice list
                await self.fetch_voices()
                
                return voice_id
                
        except Exception as e:
            raise Exception(f"ElevenLabs upload failed: {e}")

# Initialize engines
coqui_engine = CoquiTTSEngine()
elevenlabs_engine = ElevenLabsEngine()

async def generate_tts(text, user_id, engine="coqui"):
    """Main TTS generation function with error handling"""
    try:
        if engine == "elevenlabs" and elevenlabs_engine.client:
            return await elevenlabs_engine.generate_tts(text, user_id)
        elif engine == "coqui" and coqui_engine.tts:
            return await coqui_engine.generate_tts(text, user_id)
        else:
            raise Exception(f"TTS engine '{engine}' not available")
    except Exception as e:
        # If there's a voice-related error, clean up cache and try again
        if "voice" in str(e).lower() or "cache" in str(e).lower():
            clean_corrupted_cache(user_id)
        raise e
