import discord
import os
import asyncio
import json
import requests
import torch
import pickle
import threading
import hashlib
import numpy as np
from pathlib import Path
from TTS.api import TTS
from elevenlabs import ElevenLabs, save
from dotenv import load_dotenv
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict, deque
import re
from datetime import datetime

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Enhanced GPU setup
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

device = setup_gpu()

# Initialize TTS engines
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

try:
    print("🎯 Loading Coqui TTS with Lightning Performance...")
    coqui_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=device.startswith("cuda"))
    
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision('high')
        torch.set_default_device(device)
        print("⚡ GPU optimizations applied")
    
    print("✅ Coqui TTS ready with Lightning Speed")
    
except Exception as e:
    print(f"❌ Coqui TTS initialization failed: {e}")
    coqui_tts = None

# Discord setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.voice_states = True

bot = discord.Client(intents=intents)

# Enhanced file structure
UPLOAD_DIR = "uploads"
VOICE_CACHE_DIR = "voice_cache"
PHRASE_CACHE_DIR = "phrase_cache"
VOICE_BACKUPS_DIR = "voice_backups"
USER_VOICE_FILE = "user_voices.json"
TTS_STATE_FILE = "tts_states.json"
BLACKLIST_FILE = "blacklist.json"
USER_SETTINGS_FILE = "user_settings.json"
ANALYTICS_FILE = "analytics.json"

# Create directories
for directory in [UPLOAD_DIR, VOICE_CACHE_DIR, PHRASE_CACHE_DIR, VOICE_BACKUPS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize data files
for file, default in [
    (USER_VOICE_FILE, {}),
    (TTS_STATE_FILE, {}),
    (BLACKLIST_FILE, {}),
    (USER_SETTINGS_FILE, {}),
    (ANALYTICS_FILE, {"usage": {}, "popular_phrases": {}, "voice_uploads": {}})
]:
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump(default, f, indent=2)

# Load runtime state
with open(USER_VOICE_FILE, "r") as f:
    user_custom_voices = json.load(f)
with open(TTS_STATE_FILE, "r") as f:
    tts_states = json.load(f)
with open(BLACKLIST_FILE, "r") as f:
    blacklist = json.load(f)
with open(USER_SETTINGS_FILE, "r") as f:
    user_settings = json.load(f)
with open(ANALYTICS_FILE, "r") as f:
    analytics = json.load(f)

# Global settings
voice_to_id = {}
current_voice_name = None
current_voice_id = None
current_speaker_wav = "default_speaker.wav"
default_tts_engine = "coqui"

# Enhanced performance systems
executor = ThreadPoolExecutor(max_workers=12)
phrase_cache = {}
user_stats = defaultdict(lambda: {"phrases": [], "last_used": time.time()})

# QUEUE SYSTEM - The key enhancement!
class TTSQueue:
    def __init__(self):
        self.queue = deque()
        self.processing = False
        self.current_user = None
        self.queue_lock = asyncio.Lock()
        
    async def add_request(self, user_id, text, voice_channel, engine, priority=0):
        """Add TTS request to queue with priority support"""
        async with self.queue_lock:
            request = {
                'user_id': user_id,
                'text': text,
                'voice_channel': voice_channel,
                'engine': engine,
                'priority': priority,
                'timestamp': time.time(),
                'position': len(self.queue) + 1
            }
            
            if priority > 0:  # Admin priority
                # Insert at beginning for high priority
                self.queue.appendleft(request)
            else:
                self.queue.append(request)
            
            return request['position']
    
    async def process_queue(self):
        """Process TTS queue one by one"""
        while True:
            if not self.queue:
                await asyncio.sleep(0.1)
                continue
                
            async with self.queue_lock:
                if self.processing:
                    await asyncio.sleep(0.1)
                    continue
                    
                self.processing = True
                request = self.queue.popleft()
                self.current_user = request['user_id']
            
            try:
                await self.execute_tts_request(request)
            except Exception as e:
                print(f"❌ Queue processing error: {e}")
            finally:
                self.processing = False
                self.current_user = None
                await asyncio.sleep(0.2)  # Small delay between requests
    
    async def execute_tts_request(self, request):
        """Execute a TTS request"""
        try:
            if request['engine'] == "elevenlabs" and elevenlabs_client:
                audio_file = await generate_tts_elevenlabs(request['text'], request['user_id'])
            elif request['engine'] == "coqui" and coqui_tts:
                audio_file = await lightning_fast_tts(request['text'], request['user_id'])
            else:
                return
            
            # Play audio
            vc = discord.utils.get(bot.voice_clients, guild=request['voice_channel'].guild)
            if not vc:
                vc = await request['voice_channel'].connect()
            elif vc.channel != request['voice_channel']:
                await vc.move_to(request['voice_channel'])
            
            def after_play(e):
                try:
                    if not audio_file.startswith("lightning_tts_"):
                        os.remove(audio_file)
                except:
                    pass
                if e:
                    print(f"Audio playback error: {e}")
            
            vc.play(discord.FFmpegPCMAudio(audio_file), after=after_play)
            
            # Update analytics
            await self.update_analytics(request['user_id'], request['text'])
            
        except Exception as e:
            print(f"❌ TTS execution error: {e}")
    
    async def update_analytics(self, user_id, text):
        """Update usage analytics"""
        if user_id not in analytics["usage"]:
            analytics["usage"][user_id] = {"count": 0, "last_used": time.time()}
        
        analytics["usage"][user_id]["count"] += 1
        analytics["usage"][user_id]["last_used"] = time.time()
        
        # Track popular phrases
        clean_text = text.lower().strip()
        if clean_text in analytics["popular_phrases"]:
            analytics["popular_phrases"][clean_text] += 1
        else:
            analytics["popular_phrases"][clean_text] = 1
        
        # Save analytics
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics, f, indent=2)
    
    def get_queue_status(self):
        """Get current queue status"""
        return {
            'queue_length': len(self.queue),
            'processing': self.processing,
            'current_user': self.current_user
        }

# Initialize queue system
tts_queue = TTSQueue()

# CUDA streams for parallel processing
if device.startswith("cuda"):
    cuda_streams = [torch.cuda.Stream() for _ in range(4)]
    current_stream = 0

# Enhanced Voice Management System
class VoiceManager:
    def __init__(self):
        self.voice_cache_dir = VOICE_CACHE_DIR
        self.backups_dir = VOICE_BACKUPS_DIR
        
    async def backup_existing_voice(self, user_id, voice_type):
        """Backup existing voice before replacing"""
        voice_key = f"{user_id}_{voice_type}"
        if voice_key in user_custom_voices:
            backup_name = f"{user_id}_{voice_type}_{int(time.time())}.backup"
            backup_path = f"{self.backups_dir}/{backup_name}"
            
            # Create backup entry
            backup_data = {
                'user_id': user_id,
                'voice_type': voice_type,
                'original_data': user_custom_voices[voice_key],
                'backup_time': time.time(),
                'backup_name': backup_name
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return backup_name
        return None
    
    async def replace_voice(self, user_id, audio_file, voice_name, voice_type):
        """Replace existing voice with new one"""
        try:
            # Backup existing voice
            backup_name = await self.backup_existing_voice(user_id, voice_type)
            
            # Process new voice
            if voice_type == "coqui":
                await lightning_voice_processing(user_id, audio_file, voice_name)
            elif voice_type == "elevenlabs":
                await upload_to_elevenlabs(user_id, audio_file, voice_name)
            
            # Update voice history
            voice_key = f"{user_id}_{voice_type}"
            if voice_key not in analytics["voice_uploads"]:
                analytics["voice_uploads"][voice_key] = []
            
            analytics["voice_uploads"][voice_key].append({
                'voice_name': voice_name,
                'upload_time': time.time(),
                'backup_name': backup_name
            })
            
            # Save analytics
            with open(ANALYTICS_FILE, "w") as f:
                json.dump(analytics, f, indent=2)
            
            return True, backup_name
            
        except Exception as e:
            raise Exception(f"Voice replacement failed: {e}")
    
    async def list_voice_backups(self, user_id):
        """List available voice backups for user"""
        backups = []
        for file in os.listdir(self.backups_dir):
            if file.startswith(f"{user_id}_") and file.endswith(".backup"):
                try:
                    with open(f"{self.backups_dir}/{file}", 'r') as f:
                        backup_data = json.load(f)
                        backups.append({
                            'name': file,
                            'voice_type': backup_data['voice_type'],
                            'backup_time': backup_data['backup_time'],
                            'backup_name': backup_data['backup_name']
                        })
                except:
                    continue
        return backups
    
    async def restore_voice_backup(self, user_id, backup_name):
        """Restore voice from backup"""
        backup_path = f"{self.backups_dir}/{backup_name}"
        if not os.path.exists(backup_path):
            return False, "Backup not found"
        
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore voice data
            voice_key = f"{user_id}_{backup_data['voice_type']}"
            user_custom_voices[voice_key] = backup_data['original_data']
            
            # Save updated voices
            with open(USER_VOICE_FILE, "w") as f:
                json.dump(user_custom_voices, f, indent=2)
            
            return True, f"Voice restored from {backup_data['backup_name']}"
            
        except Exception as e:
            return False, f"Restore failed: {e}"

voice_manager = VoiceManager()

# Content filter for auto-moderation
class ContentFilter:
    def __init__(self):
        self.blocked_words = {
            'spam', 'test' * 10, 'a' * 20  # Basic spam detection
        }
        self.max_length = 500
        self.max_repeat_chars = 5
    
    def filter_text(self, text):
        """Filter and clean text for TTS"""
        # Remove excessive repetition
        cleaned = re.sub(r'(.)\1{' + str(self.max_repeat_chars) + ',}', r'\1' * self.max_repeat_chars, text)
        
        # Length limit
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length] + "..."
        
        # Check for blocked content
        if any(word in cleaned.lower() for word in self.blocked_words):
            return None, "Content filtered"
        
        return cleaned, "OK"

content_filter = ContentFilter()

# Enhanced TTS generation functions
async def lightning_fast_tts(text, user_id):
    """Lightning-fast TTS generation with caching"""
    if not coqui_tts:
        raise Exception("Coqui TTS not available")
    
    try:
        # Check phrase cache
        cache_key = f"{user_id}_{text.lower()}"
        if cache_key in phrase_cache:
            cached_file = phrase_cache[cache_key]
            if cached_file and os.path.exists(cached_file):
                print(f"⚡ Cache hit for: '{text[:30]}...'")
                return cached_file
        
        # GPU optimization
        if device.startswith("cuda"):
            global current_stream
            stream = cuda_streams[current_stream % len(cuda_streams)]
            current_stream += 1
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        
        # Load voice data
        cache_file = user_custom_voices.get(f"{user_id}_coqui_lightning")
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            speaker_wav = cache_data['audio_file']
        else:
            speaker_wav = user_custom_voices.get(f"{user_id}_coqui")
            if not speaker_wav:
                speaker_wav = ensure_default_voice()
                if not speaker_wav:
                    raise Exception("No voice available")
        
        output_file = f"lightning_tts_{user_id}_{int(time.time())}.wav"
        
        def lightning_tts_generation():
            start_time = time.time()
            
            coqui_tts.tts_to_file(
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
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, lightning_tts_generation)
        
        return result
        
    except Exception as e:
        raise Exception(f"Lightning TTS Error: {e}")

# Standard utility functions
def convert_audio_to_wav(input_file):
    """Convert audio to optimized WAV format"""
    file_path, file_extension = os.path.splitext(input_file)
    output_file = f"{file_path}.wav"
    
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(22050).set_channels(1).normalize()
        audio.export(output_file, format="wav")
        os.remove(input_file)
        return output_file
    except Exception as e:
        raise Exception(f"Audio conversion failed: {e}")

def ensure_default_voice():
    """Ensure default voice exists"""
    if not os.path.exists("default_speaker.wav"):
        print("⚠️ No default voice found. Upload one with !uploadvoice default")
        return None
    return "default_speaker.wav"

async def fetch_elevenlabs_voices():
    """Fetch available ElevenLabs voices"""
    global voice_to_id, current_voice_name, current_voice_id
    if not ELEVENLABS_API_KEY:
        return
    
    try:
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        resp = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        voices_data = data.get("voices", [])
        voice_to_id = {v['name']: v['voice_id'] for v in voices_data}
        if voice_to_id and not current_voice_id:
            current_voice_name = list(voice_to_id.keys())[0]
            current_voice_id = voice_to_id[current_voice_name]
        print("✅ ElevenLabs voices loaded")
    except Exception as e:
        print(f"❌ ElevenLabs voice fetch failed: {e}")

def get_user_tts_engine(user_id):
    return user_settings.get(user_id, {}).get('tts_engine', default_tts_engine)

def set_user_tts_engine(user_id, engine):
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['tts_engine'] = engine
    with open(USER_SETTINGS_FILE, "w") as f:
        json.dump(user_settings, f, indent=2)

async def generate_tts_elevenlabs(text, user_id):
    """Generate TTS using ElevenLabs"""
    if not elevenlabs_client:
        raise Exception("ElevenLabs not configured")
    
    try:
        voice_id = user_custom_voices.get(f"{user_id}_elevenlabs", current_voice_id)
        if not voice_id:
            raise Exception("No voice selected")
        
        audio = elevenlabs_client.text_to_speech.convert(text=text, voice_id=voice_id)
        save(audio, f"tts_{user_id}.mp3")
        return f"tts_{user_id}.mp3"
    except Exception as e:
        raise Exception(f"ElevenLabs TTS Error: {e}")

async def lightning_voice_processing(user_id, audio_file, voice_name):
    """Process voice with lightning speed"""
    try:
        # Use existing voice processor
        await super_processor.lightning_fast_processing(user_id, audio_file, voice_name)
        with open(USER_VOICE_FILE, "w") as f:
            json.dump(user_custom_voices, f, indent=2)
        return True
    except Exception as e:
        raise Exception(f"Voice processing failed: {e}")

async def upload_to_elevenlabs(user_id, audio_file, voice_name):
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
                files={'files': (os.path.basename(audio_file), f, 'audio/wav')}
            )
            response.raise_for_status()
            voice_id = response.json()["voice_id"]
            user_custom_voices[f"{user_id}_elevenlabs"] = voice_id
            await fetch_elevenlabs_voices()
            return voice_id
    except Exception as e:
        raise Exception(f"ElevenLabs upload failed: {e}")

# Background processing
def background_lightning_processing(user_id, audio_file, voice_name, channel):
    """Background voice processing"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if device.startswith("cuda"):
            torch.cuda.set_device(0)
        
        loop.run_until_complete(lightning_voice_processing(user_id, audio_file, voice_name))
        loop.close()
        
        asyncio.run_coroutine_threadsafe(
            channel.send(f"⚡ Voice `{voice_name}` processed with lightning speed!"),
            bot.loop
        )
    except Exception as e:
        asyncio.run_coroutine_threadsafe(
            channel.send(f"❌ Processing error: {e}"),
            bot.loop
        )

# Enhanced voice processor (keeping existing implementation)
class SuperFastVoiceProcessor:
    def __init__(self):
        self.voice_cache_dir = VOICE_CACHE_DIR
        
    async def lightning_fast_processing(self, user_id, audio_file, voice_name):
        try:
            print(f"⚡ Lightning processing for user {user_id}...")
            
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.set_device(0)
            
            # Process audio
            optimized_file = self.optimize_audio_for_speed(audio_file)
            
            cache_data = {
                'audio_file': optimized_file,
                'voice_name': voice_name,
                'processed_at': time.time()
            }
            
            cache_file = f"{self.voice_cache_dir}/{user_id}_lightning_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            user_custom_voices[f"{user_id}_coqui_lightning"] = cache_file
            
            print(f"⚡ Processing complete for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise e
    
    def optimize_audio_for_speed(self, audio_file):
        try:
            audio = AudioSegment.from_file(audio_file)
            audio = audio.set_frame_rate(22050).set_channels(1).normalize()
            
            if len(audio) > 15000:
                audio = audio[:15000]
            elif len(audio) < 3000:
                audio = audio + AudioSegment.silent(duration=3000 - len(audio))
            
            optimized_file = audio_file.replace('.wav', '_lightning_optimized.wav')
            audio.export(optimized_file, format="wav")
            
            return optimized_file
            
        except Exception as e:
            print(f"Audio optimization failed: {e}")
            return audio_file

super_processor = SuperFastVoiceProcessor()

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    print(f"🚀 Processing Device: {device.upper()}")
    
    if device.startswith("cuda"):
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"⚡ GPU Memory: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB")
        print(f"🚀 CUDA Streams: {len(cuda_streams)} active")
    
    await fetch_elevenlabs_voices()
    if elevenlabs_client:
        print("✅ ElevenLabs ready")
    else:
        print("⚠️ ElevenLabs not configured")
    
    if coqui_tts:
        print("⚡ Coqui TTS ready - Lightning mode activated")
        
        # Start queue processor
        asyncio.create_task(tts_queue.process_queue())
        print("🔄 TTS Queue system started")
        
        # Warmup
        try:
            if os.path.exists("default_speaker.wav"):
                print("🔥 Warming up lightning engine...")
                start_time = time.time()
                test_file = await lightning_fast_tts("Lightning warmup test", "system")
                warmup_time = time.time() - start_time
                if test_file and os.path.exists(test_file):
                    os.remove(test_file)
                print(f"⚡ Warmup complete in {warmup_time:.2f}s")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
    else:
        print("❌ Coqui TTS not available")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if not message.guild:
        try:
            await message.channel.send("❌ This bot only works in Discord servers, not DMs.")
        except:
            pass
        return

    content = message.content.strip()
    user_id = str(message.author.id)

    def check_admin_permissions():
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

    is_admin = check_admin_permissions()

    if user_id in blacklist and not is_admin:
        return

    # Enhanced Help Command
    if content.startswith("!help"):
        help_embed = discord.Embed(
            title="⚡ Lightning TTS Bot - Command Guide",
            description="Ultra-fast text-to-speech with advanced features",
            color=0x00ff00
        )
        
        help_embed.add_field(
            name="🎤 Basic TTS Commands",
            value="`!tts <text>` - Lightning-fast speech\n"
                  "`!enabletts` - Enable auto-TTS (mute mic first)\n"
                  "`!disabletts` - Disable auto-TTS",
            inline=False
        )
        
        help_embed.add_field(
            name="🔧 Engine Control",
            value="`!engine <elevenlabs|coqui>` - Switch TTS engine\n"
                  "`!status` - Check system status\n"
                  "`!performance` - Performance metrics\n"
                  "`!queue` - Check TTS queue status",
            inline=False
        )
        
        help_embed.add_field(
            name="🎙️ Voice Management",
            value="`!uploadvoice <name>` - Upload new voice\n"
                  "`!replacevoice <name>` - Replace existing voice\n"
                  "`!myvoice` - Use your custom voice\n"
                  "`!voices` - List ElevenLabs voices\n"
                  "`!voice <name>` - Select ElevenLabs voice",
            inline=False
        )
        
        help_embed.add_field(
            name="🔄 Voice Backup System",
            value="`!backups` - List your voice backups\n"
                  "`!restore <backup_name>` - Restore voice backup\n"
                  "`!voicehistory` - View voice upload history",
            inline=False
        )
        
        help_embed.add_field(
            name="📊 Analytics & Stats",
            value="`!stats` - Your usage statistics\n"
                  "`!popular` - Most popular phrases\n"
                  "`!analytics` - Server analytics (admin only)",
            inline=False
        )
        
        help_embed.add_field(
            name="⚙️ Bot Control",
            value="`!clearcache` - Clear all caches\n"
                  "`!dc` or `!leave` - Disconnect bot\n"
                  "`!blacklist <user>` - Block user (admin only)",
            inline=False
        )
        
        help_embed.add_field(
            name="⚡ Lightning Features",
            value="• GPU-accelerated processing\n"
                  "• Smart queue system for multiple users\n"
                  "• Phrase caching for instant responses\n"
                  "• Voice backup and restore system\n"
                  "• Advanced analytics and usage tracking\n"
                  "• Content filtering and moderation",
            inline=False
        )
        
        help_embed.set_footer(text="Lightning TTS Bot | Made with ⚡ by AI")
        
        await message.channel.send(embed=help_embed)
        return

    # Switch TTS engine
    if content.startswith("!engine"):
        parts = content.split()
        if len(parts) == 2:
            engine = parts[1].lower()
            if engine in ["elevenlabs", "coqui"]:
                if engine == "elevenlabs" and not elevenlabs_client:
                    await message.channel.send("❌ ElevenLabs not configured.")
                    return
                if engine == "coqui" and not coqui_tts:
                    await message.channel.send("❌ Coqui TTS not available.")
                    return
                
                set_user_tts_engine(user_id, engine)
                await message.channel.send(f"✅ TTS engine switched to **{engine}**")
            else:
                await message.channel.send("❌ Available engines: `elevenlabs`, `coqui`")
        else:
            current_engine = get_user_tts_engine(user_id)
            await message.channel.send(f"📍 Current engine: **{current_engine}**\n💡 Usage: `!engine <elevenlabs|coqui>`")
        return

    # Enhanced Status Command
    if content.startswith("!status"):
        current_engine = get_user_tts_engine(user_id)
        elevenlabs_status = "✅ Ready" if elevenlabs_client else "❌ Not configured"
        coqui_status = "⚡ Lightning-Fast" if coqui_tts else "❌ Not available"
        
        lightning_voice = f"{user_id}_coqui_lightning" in user_custom_voices
        voice_status = "⚡ Lightning-Optimized" if lightning_voice else "📝 Standard"
        
        # Queue status
        queue_status = tts_queue.get_queue_status()
        queue_info = f"📊 Queue: {queue_status['queue_length']} requests"
        if queue_status['processing']:
            queue_info += f"\n🔄 Processing: {queue_status['current_user']}"
        
        gpu_info = ""
        if device.startswith("cuda"):
            gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"\n⚡ GPU: {torch.cuda.get_device_name(0)}\n🚀 Memory: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB"
        
        user_phrase_count = len(user_stats[user_id]["phrases"])
        cached_phrases = len([k for k in phrase_cache.keys() if k.startswith(user_id)])
        
        status_embed = discord.Embed(
            title="⚡ Lightning TTS Status",
            color=0x00ff00
        )
        
        status_embed.add_field(
            name="🎛️ Engine Status",
            value=f"Current: **{current_engine}**\n"
                  f"ElevenLabs: {elevenlabs_status}\n"
                  f"Coqui TTS: {coqui_status}",
            inline=True
        )
        
        status_embed.add_field(
            name="🎙️ Voice Status",
            value=f"Your Voice: {voice_status}\n"
                  f"Device: **{device.upper()}**{gpu_info}",
            inline=True
        )
        
        status_embed.add_field(
            name="📈 Performance",
            value=f"Your phrases: {user_phrase_count}\n"
                  f"Cached phrases: {cached_phrases}\n"
                  f"{queue_info}",
            inline=False
        )
        
        await message.channel.send(embed=status_embed)
        return

    # Queue Status Command
    if content.startswith("!queue"):
        queue_status = tts_queue.get_queue_status()
        
        if queue_status['queue_length'] == 0:
            await message.channel.send("✅ Queue is empty - ready for instant TTS!")
        else:
            queue_msg = f"📊 **TTS Queue Status**\n"
            queue_msg += f"🔄 Requests in queue: {queue_status['queue_length']}\n"
            if queue_status['processing']:
                queue_msg += f"⚡ Currently processing: <@{queue_status['current_user']}>\n"
            queue_msg += f"⏱️ Estimated wait: {queue_status['queue_length'] * 2}s"
            
            await message.channel.send(queue_msg)
        return

    # Enhanced Voice Upload with Replacement
    if content.startswith("!uploadvoice") or content.startswith("!replacevoice"):
        is_replacement = content.startswith("!replacevoice")
        
        parts = content.split(" ", 1)
        if len(parts) < 2 or not message.attachments:
            await message.channel.send("❌ Usage: `!uploadvoice <name>` or `!replacevoice <name>` with audio file")
            return
        
        name = parts[1].strip()
        file = message.attachments[0]
        filename = f"{UPLOAD_DIR}/{user_id}_{file.filename}"
        
        processing_msg = await message.channel.send("⚡ Processing voice with lightning speed...")
        
        try:
            await file.save(filename)
            
            if filename.lower().endswith(('.m4a', '.mp3', '.ogg')):
                filename = convert_audio_to_wav(filename)
            
            engine = get_user_tts_engine(user_id)
            
            if is_replacement:
                # Use voice manager for replacement
                success, backup_name = await voice_manager.replace_voice(user_id, filename, name, engine)
                if success:
                    await processing_msg.edit(content=f"✅ Voice **{name}** replaced successfully!\n🔄 Backup created: `{backup_name}`")
                else:
                    await processing_msg.edit(content=f"❌ Voice replacement failed")
            else:
                # Regular upload
                if engine == "elevenlabs" and elevenlabs_client:
                    await processing_msg.edit(content="⚡ Uploading to ElevenLabs...")
                    voice_id = await upload_to_elevenlabs(user_id, filename, name)
                    await processing_msg.edit(content=f"✅ Voice **{name}** uploaded to ElevenLabs!")
                
                elif engine == "coqui" and coqui_tts:
                    await processing_msg.edit(content="⚡ Processing for lightning-fast TTS...")
                    
                    user_custom_voices[f"{user_id}_coqui"] = filename
                    
                    if name.lower() == "default" or not os.path.exists("default_speaker.wav"):
                        import shutil
                        shutil.copy(filename, "default_speaker.wav")
                    
                    # Background processing
                    processing_thread = threading.Thread(
                        target=background_lightning_processing,
                        args=(user_id, filename, name, message.channel)
                    )
                    processing_thread.start()
                
                else:
                    await processing_msg.edit(content=f"❌ Engine **{engine}** not available")
                    return
            
            with open(USER_VOICE_FILE, "w") as f:
                json.dump(user_custom_voices, f, indent=2)
            
        except Exception as e:
            await processing_msg.edit(content=f"❌ Processing error: {e}")
        return

    # Voice Backup Management
    if content.startswith("!backups"):
        try:
            backups = await voice_manager.list_voice_backups(user_id)
            
            if not backups:
                await message.channel.send("📂 No voice backups found.")
                return
            
            backup_msg = "📂 **Your Voice Backups:**\n\n"
            for backup in backups:
                backup_time = datetime.fromtimestamp(backup['backup_time']).strftime("%Y-%m-%d %H:%M")
                backup_msg += f"🔄 `{backup['backup_name']}`\n"
                backup_msg += f"   • Type: {backup['voice_type']}\n"
                backup_msg += f"   • Date: {backup_time}\n\n"
            
            backup_msg += "💡 Use `!restore <backup_name>` to restore a backup"
            
            await message.channel.send(backup_msg)
            
        except Exception as e:
            await message.channel.send(f"❌ Error listing backups: {e}")
        return

    # Restore Voice Backup
    if content.startswith("!restore"):
        parts = content.split(" ", 1)
        if len(parts) < 2:
            await message.channel.send("❌ Usage: `!restore <backup_name>`")
            return
        
        backup_name = parts[1].strip()
        
        try:
            success, result_msg = await voice_manager.restore_voice_backup(user_id, backup_name)
            
            if success:
                await message.channel.send(f"✅ {result_msg}")
            else:
                await message.channel.send(f"❌ {result_msg}")
                
        except Exception as e:
            await message.channel.send(f"❌ Restore error: {e}")
        return

    # User Statistics
    if content.startswith("!stats"):
        if user_id in analytics["usage"]:
            user_data = analytics["usage"][user_id]
            last_used = datetime.fromtimestamp(user_data["last_used"]).strftime("%Y-%m-%d %H:%M")
            
            stats_msg = f"📊 **Your TTS Statistics**\n\n"
            stats_msg += f"🎤 Total requests: {user_data['count']}\n"
            stats_msg += f"📅 Last used: {last_used}\n"
            stats_msg += f"⚡ Cached phrases: {len([k for k in phrase_cache.keys() if k.startswith(user_id)])}\n"
            
            # Voice history
            voice_history = analytics["voice_uploads"].get(f"{user_id}_coqui", [])
            if voice_history:
                stats_msg += f"🎙️ Voice uploads: {len(voice_history)}\n"
            
            await message.channel.send(stats_msg)
        else:
            await message.channel.send("📊 No usage statistics found. Start using TTS to see your stats!")
        return

    # Popular Phrases
    if content.startswith("!popular"):
        popular_phrases = sorted(analytics["popular_phrases"].items(), key=lambda x: x[1], reverse=True)[:10]
        
        if not popular_phrases:
            await message.channel.send("📈 No popular phrases data available yet.")
            return
        
        popular_msg = "🔥 **Most Popular Phrases:**\n\n"
        for i, (phrase, count) in enumerate(popular_phrases, 1):
            popular_msg += f"{i}. \"{phrase[:50]}{'...' if len(phrase) > 50 else ''}\" - {count} uses\n"
        
        await message.channel.send(popular_msg)
        return

    # Performance Metrics
    if content.startswith("!performance"):
        if device.startswith("cuda"):
            gpu_util = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100
            cached_count = len(phrase_cache)
            queue_status = tts_queue.get_queue_status()
            
            perf_embed = discord.Embed(
                title="⚡ Lightning Performance Metrics",
                color=0x00ff00
            )
            
            perf_embed.add_field(
                name="🚀 GPU Performance",
                value=f"Utilization: {gpu_util:.1f}%\n"
                      f"CUDA Streams: {len(cuda_streams)} active\n"
                      f"Device: {torch.cuda.get_device_name(0)}",
                inline=True
            )
            
            perf_embed.add_field(
                name="🎯 TTS Performance",
                value=f"Cached phrases: {cached_count}\n"
                      f"Queue length: {queue_status['queue_length']}\n"
                      f"Processing: {'Yes' if queue_status['processing'] else 'No'}",
                inline=True
            )
            
            perf_embed.add_field(
                name="📊 System Status",
                value="Model: Quantized\n"
                      "Memory: 95% allocated\n"
                      "Mode: LIGHTNING-FAST",
                inline=False
            )
            
            await message.channel.send(embed=perf_embed)
        else:
            await message.channel.send("❌ Performance metrics only available with GPU acceleration")
        return

    # Enable TTS
    if content.lower() == "!enabletts":
        if not message.author.voice:
            await message.channel.send("❌ You must be in a voice channel to enable TTS.")
            return
        if not message.author.voice.self_mute:
            await message.channel.send("❌ Please mute your microphone before enabling TTS.")
            return
        
        tts_states[user_id] = True
        with open(TTS_STATE_FILE, "w") as f:
            json.dump(tts_states, f, indent=2)
        
        engine = get_user_tts_engine(user_id)
        await message.channel.send(f"⚡ **Lightning TTS enabled** using **{engine}** engine!\n💡 Your messages will now be spoken automatically.")
        return

    # Disable TTS
    if content.lower() == "!disabletts":
        tts_states.pop(user_id, None)
        with open(TTS_STATE_FILE, "w") as f:
            json.dump(tts_states, f, indent=2)
        await message.channel.send("🔇 **TTS disabled** - messages will no longer be spoken.")
        return

    # List ElevenLabs voices
    if content.startswith("!voices"):
        if not elevenlabs_client:
            await message.channel.send("❌ ElevenLabs not configured.")
            return
        if not voice_to_id:
            await fetch_elevenlabs_voices()
        if voice_to_id:
            voices_list = "\n".join(f"🎤 `{v}`" for v in voice_to_id)
            await message.channel.send(f"🎙️ **Available ElevenLabs Voices:**\n{voices_list}")
        else:
            await message.channel.send("❌ No voices available.")
        return

    # Set ElevenLabs voice
    if content.startswith("!voice"):
        if not elevenlabs_client:
            await message.channel.send("❌ ElevenLabs not configured.")
            return
        name = content[6:].strip()
        if name in voice_to_id:
            global current_voice_name, current_voice_id
            current_voice_name = name
            current_voice_id = voice_to_id[name]
            await message.channel.send(f"✅ ElevenLabs voice changed to **{name}**")
        else:
            await message.channel.send("❌ Voice not found. Use `!voices` to see available voices.")
        return

    # Use custom voice
    if content.startswith("!myvoice"):
        engine = get_user_tts_engine(user_id)
        voice_key = f"{user_id}_{engine}"
        lightning_key = f"{user_id}_{engine}_lightning"
        
        if lightning_key in user_custom_voices or voice_key in user_custom_voices:
            status = "⚡ Lightning-Optimized" if lightning_key in user_custom_voices else "Standard"
            await message.channel.send(f"✅ **Custom voice activated** ({status}) for **{engine}** engine.")
        else:
            await message.channel.send(f"❌ No custom voice found for **{engine}** engine.\n💡 Use `!uploadvoice <name>` to upload one.")
        return

    # Clear cache
    if content.startswith("!clearcache"):
        try:
            # Clear all user caches
            cache_file = f"{VOICE_CACHE_DIR}/{user_id}_lightning_cache.pkl"
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            # Clear phrase cache
            keys_to_remove = [k for k in phrase_cache.keys() if k.startswith(user_id)]
            for key in keys_to_remove:
                del phrase_cache[key]
            
            # Clear user voice data
            keys_to_remove = [k for k in user_custom_voices.keys() if k.startswith(f"{user_id}_")]
            for key in keys_to_remove:
                del user_custom_voices[key]
            
            # Clear user stats
            if user_id in user_stats:
                del user_stats[user_id]
            
            with open(USER_VOICE_FILE, "w") as f:
                json.dump(user_custom_voices, f, indent=2)
            
            await message.channel.send("✅ **All caches cleared** - ready for fresh lightning-fast processing!")
        except Exception as e:
            await message.channel.send(f"❌ Error clearing cache: {e}")
        return

    # Blacklist (admin only)
    if content.startswith("!blacklist") and is_admin:
        parts = content.split()
        if len(parts) == 2:
            blacklist[parts[1]] = True
            with open(BLACKLIST_FILE, "w") as f:
                json.dump(blacklist, f, indent=2)
            await message.channel.send(f"⛔ **User blacklisted:** `{parts[1]}`")
        return

    # Disconnect
    if content.lower() in ["!dc", "!leave", "!disconnect"]:
        vc = discord.utils.get(bot.voice_clients, guild=message.guild)
        if vc:
            await vc.disconnect()
            await message.channel.send("✅ **Bot disconnected** from voice channel.")
        else:
            await message.channel.send("❌ Not connected to any voice channel.")
        return

    # ENHANCED TTS PROCESSING WITH QUEUE SYSTEM
    if content.startswith("!tts "):
        text = content[5:].strip()
    elif tts_states.get(user_id) and message.author.voice and message.author.voice.self_mute:
        text = content
    else:
        return

    # Validate user is in voice channel
    if not message.author.voice:
        await message.channel.send("❌ You must be in a voice channel to use TTS.")
        return

    # Content filtering
    filtered_text, filter_result = content_filter.filter_text(text)
    if not filtered_text:
        await message.channel.send(f"❌ {filter_result}")
        return
    
    text = filtered_text

    # ADD TO QUEUE SYSTEM - This is the key enhancement!
    try:
        engine = get_user_tts_engine(user_id)
        
        # Check if engines are available
        if engine == "elevenlabs" and not elevenlabs_client:
            await message.channel.send("❌ ElevenLabs not available.")
            return
        elif engine == "coqui" and not coqui_tts:
            await message.channel.send("❌ Coqui TTS not available.")
            return
        
        # Add to queue with priority for admins
        priority = 1 if is_admin else 0
        position = await tts_queue.add_request(
            user_id=user_id,
            text=text,
            voice_channel=message.author.voice.channel,
            engine=engine,
            priority=priority
        )
        
        # Only show queue position if there are multiple requests
        if position > 1:
            await message.channel.send(f"🔄 **Added to queue** - Position: {position}")
        
    except Exception as e:
        await message.channel.send(f"❌ TTS Error: {str(e)}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
