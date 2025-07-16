"""
Voice Manager Module for Lightning Discord TTS Bot
Handles voice uploads, processing, backups, and management
"""

import os
import json
import time
import pickle
import asyncio
from pathlib import Path
from pydub import AudioSegment
from datetime import datetime

from .utils import (
    UPLOAD_DIR,
    VOICE_CACHE_DIR,
    VOICE_BACKUPS_DIR,
    USER_VOICE_FILE,
    ANALYTICS_FILE,
    load_json_file,
    save_json_file,
    setup_gpu,
    clean_corrupted_cache
)

device = setup_gpu()

class VoiceProcessor:
    """Voice processing and optimization with robust error handling"""
    
    def __init__(self):
        self.voice_cache_dir = VOICE_CACHE_DIR
        
    def convert_audio_to_wav(self, input_file):
        """Convert audio to optimized WAV format"""
        file_path = Path(input_file)
        output_file = file_path.with_suffix('.wav')
        
        try:
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_frame_rate(22050).set_channels(1).normalize()
            audio.export(output_file, format="wav")
            
            # Remove original if different
            if input_file != str(output_file):
                file_path.unlink()
            
            return str(output_file)
            
        except Exception as e:
            raise Exception(f"Audio conversion failed: {e}")
    
    def optimize_audio_for_speed(self, audio_file):
        """Optimize audio for maximum TTS speed"""
        try:
            audio = AudioSegment.from_file(audio_file)
            
            # Speed-optimized settings
            audio = audio.set_frame_rate(22050).set_channels(1).normalize()
            
            # Optimal length for TTS
            if len(audio) > 15000:  # 15 seconds max
                audio = audio[:15000]
            elif len(audio) < 3000:  # 3 seconds min
                audio = audio + AudioSegment.silent(duration=3000 - len(audio))
            
            optimized_file = str(Path(audio_file).with_name(
                f"{Path(audio_file).stem}_optimized.wav"
            ))
            
            audio.export(optimized_file, format="wav")
            return optimized_file
            
        except Exception as e:
            print(f"Audio optimization failed: {e}")
            return audio_file
    
    async def lightning_fast_processing(self, user_id, audio_file, voice_name):
        """Lightning-fast voice processing with robust error handling"""
        try:
            print(f"⚡ Lightning processing for user {user_id}...")
            
            # Clean up any existing corrupted cache first
            clean_corrupted_cache(user_id)
            
            # GPU memory optimization
            if device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
                torch.cuda.set_device(0)
            
            # Process audio
            optimized_file = self.optimize_audio_for_speed(audio_file)
            
            # Create cache data with validation
            cache_data = {
                'audio_file': optimized_file,
                'voice_name': voice_name,
                'user_id': user_id,
                'processed_at': time.time(),
                'device': device,
                'version': '1.0'  # Add version for future compatibility
            }
            
            # Save to cache with error handling
            cache_file = self.voice_cache_dir / f"{user_id}_lightning_cache.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
                # Validate the cache was written correctly
                with open(cache_file, 'rb') as f:
                    test_data = pickle.load(f)
                    
                if test_data.get('audio_file') != optimized_file:
                    raise Exception("Cache validation failed")
                    
            except Exception as e:
                print(f"❌ Cache save failed: {e}")
                # Clean up failed cache
                if cache_file.exists():
                    cache_file.unlink()
                raise e
            
            # Update user voices
            user_voices = load_json_file(USER_VOICE_FILE)
            user_voices[f"{user_id}_coqui_lightning"] = str(cache_file)
            user_voices[f"{user_id}_coqui"] = optimized_file
            save_json_file(USER_VOICE_FILE, user_voices)
            
            print(f"⚡ Processing complete for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            # Clean up any partial data
            clean_corrupted_cache(user_id)
            raise e

class VoiceBackupManager:
    """Voice backup and restore system"""
    
    def __init__(self):
        self.backups_dir = VOICE_BACKUPS_DIR
        
    async def backup_existing_voice(self, user_id, voice_type):
        """Backup existing voice before replacing"""
        user_voices = load_json_file(USER_VOICE_FILE)
        voice_key = f"{user_id}_{voice_type}"
        
        if voice_key in user_voices:
            backup_name = f"{user_id}_{voice_type}_{int(time.time())}.backup"
            backup_path = self.backups_dir / backup_name
            
            backup_data = {
                'user_id': user_id,
                'voice_type': voice_type,
                'original_data': user_voices[voice_key],
                'backup_time': time.time(),
                'backup_name': backup_name
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return backup_name
        return None
    
    async def list_voice_backups(self, user_id):
        """List available voice backups for user"""
        backups = []
        
        for backup_file in self.backups_dir.glob(f"{user_id}_*.backup"):
            try:
                with open(backup_file, 'r') as f:
                    backup_data = json.load(f)
                    backups.append({
                        'name': backup_file.name,
                        'voice_type': backup_data['voice_type'],
                        'backup_time': backup_data['backup_time'],
                        'backup_name': backup_data['backup_name']
                    })
            except:
                continue
        
        return backups
    
    async def restore_voice_backup(self, user_id, backup_name):
        """Restore voice from backup"""
        backup_path = self.backups_dir / backup_name
        
        if not backup_path.exists():
            return False, "Backup not found"
        
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore voice data
            user_voices = load_json_file(USER_VOICE_FILE)
            voice_key = f"{user_id}_{backup_data['voice_type']}"
            user_voices[voice_key] = backup_data['original_data']
            save_json_file(USER_VOICE_FILE, user_voices)
            
            return True, f"Voice restored from {backup_data['backup_name']}"
            
        except Exception as e:
            return False, f"Restore failed: {e}"

class VoiceManager:
    """Main voice management system"""
    
    def __init__(self):
        self.processor = VoiceProcessor()
        self.backup_manager = VoiceBackupManager()
        
    async def upload_voice(self, user_id, file_attachment, voice_name):
        """Handle voice upload with error handling"""
        try:
            # Save uploaded file
            filename = UPLOAD_DIR / f"{user_id}_{file_attachment.filename}"
            await file_attachment.save(filename)
            
            # Convert to WAV if needed
            if filename.suffix.lower() in ['.m4a', '.mp3', '.ogg']:
                filename = Path(self.processor.convert_audio_to_wav(str(filename)))
            
            return str(filename)
            
        except Exception as e:
            raise Exception(f"Voice upload failed: {e}")
    
    async def replace_voice(self, user_id, audio_file, voice_name, voice_type):
        """Replace existing voice with backup"""
        try:
            # Backup existing voice
            backup_name = await self.backup_manager.backup_existing_voice(user_id, voice_type)
            
            # Process new voice
            if voice_type == "coqui":
                await self.process_coqui_voice(user_id, audio_file, voice_name)
            elif voice_type == "elevenlabs":
                from .tts_engine import elevenlabs_engine
                await elevenlabs_engine.upload_voice(user_id, audio_file, voice_name)
            
            # Update analytics
            await self._update_voice_analytics(user_id, voice_name, voice_type, backup_name)
            
            return True, backup_name
            
        except Exception as e:
            raise Exception(f"Voice replacement failed: {e}")
    
    async def process_coqui_voice(self, user_id, audio_file, voice_name):
        """Process voice for Coqui TTS"""
        await self.processor.lightning_fast_processing(user_id, audio_file, voice_name)
    
    async def get_user_voice_info(self, user_id):
        """Get user's voice information"""
        user_voices = load_json_file(USER_VOICE_FILE)
        
        info = {
            'coqui_voice': None,
            'coqui_lightning': None,
            'elevenlabs_voice': None
        }
        
        # Check for voices
        if f"{user_id}_coqui" in user_voices:
            info['coqui_voice'] = user_voices[f"{user_id}_coqui"]
        
        if f"{user_id}_coqui_lightning" in user_voices:
            info['coqui_lightning'] = user_voices[f"{user_id}_coqui_lightning"]
        
        if f"{user_id}_elevenlabs" in user_voices:
            info['elevenlabs_voice'] = user_voices[f"{user_id}_elevenlabs"]
        
        return info
    
    async def _update_voice_analytics(self, user_id, voice_name, voice_type, backup_name):
        """Update voice upload analytics"""
        analytics = load_json_file(ANALYTICS_FILE, {
            "usage": {},
            "popular_phrases": {},
            "voice_uploads": {}
        })
        
        voice_key = f"{user_id}_{voice_type}"
        if voice_key not in analytics["voice_uploads"]:
            analytics["voice_uploads"][voice_key] = []
        
        analytics["voice_uploads"][voice_key].append({
            'voice_name': voice_name,
            'upload_time': time.time(),
            'backup_name': backup_name
        })
        
        save_json_file(ANALYTICS_FILE, analytics)

# Background voice processing using asyncio tasks
async def process_voice_background(user_id, audio_file, voice_name, channel):
    """Process voice in background using main event loop"""
    try:
        voice_manager = VoiceManager()
        await voice_manager.process_coqui_voice(user_id, audio_file, voice_name)
        await channel.send(f"⚡ Voice `{voice_name}` processed with lightning speed!")
        
    except Exception as e:
        await channel.send(f"❌ Processing error: {e}")

# Initialize voice manager
voice_manager = VoiceManager()
