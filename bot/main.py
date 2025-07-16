"""
Main Discord Bot Handler for Lightning Discord TTS Bot
Handles Discord events, commands, and integrates all modules
"""

import discord
import asyncio
from pathlib import Path
from datetime import datetime

from .utils import (
    DISCORD_TOKEN,
    setup_directories,
    load_json_file,
    save_json_file,
    check_admin_permissions,
    content_filter,
    TTS_STATE_FILE,
    USER_SETTINGS_FILE,
    BLACKLIST_FILE,
    ANALYTICS_FILE,
    DEFAULT_TTS_ENGINE,
    clean_corrupted_cache
)

from .tts_engine import coqui_engine, elevenlabs_engine
from .voice_manager import voice_manager, process_voice_background
from .queue_system import queue_manager

# Discord setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.voice_states = True

class LightningTTSBot(discord.Client):
    """Lightning TTS Bot main class with comprehensive error handling"""
    
    def __init__(self):
        super().__init__(intents=intents)
        self.setup_complete = False
        
    async def setup_hook(self):
        """Setup hook called when bot starts"""
        print("🔧 Setting up Lightning TTS Bot...")
        setup_directories()
        
        # Pass bot instance to queue manager
        queue_manager.set_bot_instance(self)
        
        await queue_manager.start()
        self.setup_complete = True
        print("✅ Setup complete")

    async def on_ready(self):
        """Called when bot is ready"""
        print(f"✅ Logged in as {self.user}")
        print(f"🚀 Lightning TTS Bot ready!")
        
        # Initialize ElevenLabs after event loop is running
        await elevenlabs_engine.initialize()
        
        # Display engine status
        if elevenlabs_engine.client:
            print("✅ ElevenLabs ready")
        else:
            print("⚠️ ElevenLabs not configured")
        
        if coqui_engine.tts:
            print("⚡ Coqui TTS ready with Lightning acceleration")
            
            # Warmup test
            try:
                if Path("default_speaker.wav").exists():
                    print("🔥 Warming up lightning engine...")
                    from .tts_engine import generate_tts
                    test_file = await generate_tts("Lightning warmup test", "system")
                    if test_file and Path(test_file).exists():
                        Path(test_file).unlink()
                    print("⚡ Warmup complete")
            except Exception as e:
                print(f"⚠️ Warmup failed: {e}")
        else:
            print("❌ Coqui TTS not available")

    async def on_message(self, message):
        """Handle incoming messages"""
        if message.author == self.user:
            return

        # Only process messages from guilds (servers)
        if not message.guild:
            try:
                await message.channel.send("❌ This bot only works in Discord servers, not DMs.")
            except:
                pass
            return

        await self.process_message(message)

    async def process_message(self, message):
        """Process message for commands or TTS"""
        content = message.content.strip()
        user_id = str(message.author.id)
        is_admin = check_admin_permissions(message)

        # Check blacklist
        blacklist = load_json_file(BLACKLIST_FILE)
        if user_id in blacklist and not is_admin:
            return

        # Handle commands
        if content.startswith("!"):
            await self.handle_command(message, content, user_id, is_admin)
            return

        # Handle auto-TTS
        await self.handle_auto_tts(message, content, user_id)

    async def handle_command(self, message, content, user_id, is_admin):
        """Handle bot commands"""
        
        # Help command
        if content.startswith("!help"):
            await self.send_help(message)
            return

        # Engine commands
        if content.startswith("!engine"):
            await self.handle_engine_command(message, content, user_id)
            return

        # Status commands
        if content.startswith("!status"):
            await self.handle_status_command(message, user_id)
            return

        if content.startswith("!performance"):
            await self.handle_performance_command(message)
            return

        # TTS commands
        if content.startswith("!tts "):
            await self.handle_tts_command(message, content[5:].strip(), user_id)
            return

        if content.lower() == "!enabletts":
            await self.handle_enable_tts(message, user_id)
            return

        if content.lower() == "!disabletts":
            await self.handle_disable_tts(message, user_id)
            return

        # Voice management commands
        if content.startswith("!uploadvoice") or content.startswith("!replacevoice"):
            await self.handle_voice_upload(message, content, user_id)
            return

        if content.startswith("!myvoice"):
            await self.handle_my_voice(message, user_id)
            return

        if content.startswith("!voices"):
            await self.handle_list_voices(message)
            return

        if content.startswith("!voice"):
            await self.handle_set_voice(message, content, user_id)
            return

        # Voice backup commands
        if content.startswith("!backups"):
            await self.handle_list_backups(message, user_id)
            return

        if content.startswith("!restore"):
            await self.handle_restore_backup(message, content, user_id)
            return

        # Queue commands
        if content.startswith("!queue"):
            await self.handle_queue_command(message)
            return

        # Analytics commands
        if content.startswith("!stats"):
            await self.handle_stats_command(message, user_id)
            return

        if content.startswith("!popular"):
            await self.handle_popular_command(message)
            return

        # Cache management
        if content.startswith("!clearcache"):
            await self.handle_clear_cache(message, user_id)
            return
        
        if content.startswith("!clearmycache"):
            await self.handle_clear_my_cache(message, user_id)
            return

        # Admin commands
        if content.startswith("!blacklist") and is_admin:
            await self.handle_blacklist_command(message, content)
            return

        # Bot control commands
        if content.lower() in ["!dc", "!leave", "!disconnect"]:
            await self.handle_disconnect(message)
            return

    async def handle_auto_tts(self, message, content, user_id):
        """Handle automatic TTS for enabled users"""
        tts_states = load_json_file(TTS_STATE_FILE)
        
        if (user_id in tts_states and 
            message.author.voice and 
            message.author.voice.self_mute):
            
            await self.process_tts_request(message, content, user_id)

    async def process_tts_request(self, message, text, user_id):
        """Process TTS request through queue system"""
        # Validate user is in voice channel
        if not message.author.voice:
            await message.channel.send("❌ You must be in a voice channel to use TTS.")
            return

        # Content filtering
        filtered_text, filter_result = content_filter.filter_text(text)
        if not filtered_text:
            await message.channel.send(f"❌ {filter_result}")
            return

        # Get user's TTS engine
        user_settings = load_json_file(USER_SETTINGS_FILE)
        engine = user_settings.get(user_id, {}).get('tts_engine', DEFAULT_TTS_ENGINE)

        # Validate engine availability
        if engine == "elevenlabs" and not elevenlabs_engine.client:
            await message.channel.send("❌ ElevenLabs not available.")
            return
        elif engine == "coqui" and not coqui_engine.tts:
            await message.channel.send("❌ Coqui TTS not available.")
            return

        # Add to queue
        try:
            position = await queue_manager.add_tts_request(
                user_id=user_id,
                text=filtered_text,
                voice_channel=message.author.voice.channel,
                engine=engine,
                message_author=message.author
            )

            # Only show queue position if there are multiple requests
            if position > 1:
                await message.channel.send(f"🔄 **Added to queue** - Position: {position}")

        except Exception as e:
            await message.channel.send(f"❌ TTS Error: {str(e)}")

    async def send_help(self, message):
        """Send help embed"""
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
                  "`!voices` - List ElevenLabs voices",
            inline=False
        )
        
        help_embed.add_field(
            name="🔄 Voice Backup & Cache",
            value="`!backups` - List voice backups\n"
                  "`!restore <backup_name>` - Restore voice backup\n"
                  "`!clearmycache` - Clear your voice cache\n"
                  "`!clearcache` - Clear all caches (admin)",
            inline=False
        )
        
        help_embed.add_field(
            name="📊 Analytics & Stats",
            value="`!stats` - Your usage statistics\n"
                  "`!popular` - Most popular phrases\n"
                  "`!queue` - Queue status",
            inline=False
        )
        
        help_embed.set_footer(text="Lightning TTS Bot | Made with ⚡ by C0deRatoR")
        await message.channel.send(embed=help_embed)

    async def handle_engine_command(self, message, content, user_id):
        """Handle engine switching"""
        parts = content.split()
        user_settings = load_json_file(USER_SETTINGS_FILE)
        
        if len(parts) == 2:
            engine = parts[1].lower()
            if engine in ["elevenlabs", "coqui"]:
                if engine == "elevenlabs" and not elevenlabs_engine.client:
                    await message.channel.send("❌ ElevenLabs not configured.")
                    return
                if engine == "coqui" and not coqui_engine.tts:
                    await message.channel.send("❌ Coqui TTS not available.")
                    return
                
                if user_id not in user_settings:
                    user_settings[user_id] = {}
                user_settings[user_id]['tts_engine'] = engine
                save_json_file(USER_SETTINGS_FILE, user_settings)
                
                await message.channel.send(f"✅ TTS engine switched to **{engine}**")
            else:
                await message.channel.send("❌ Available engines: `elevenlabs`, `coqui`")
        else:
            current_engine = user_settings.get(user_id, {}).get('tts_engine', DEFAULT_TTS_ENGINE)
            await message.channel.send(f"📍 Current engine: **{current_engine}**")

    async def handle_tts_command(self, message, text, user_id):
        """Handle direct TTS command"""
        await self.process_tts_request(message, text, user_id)

    async def handle_enable_tts(self, message, user_id):
        """Handle enable TTS command"""
        if not message.author.voice:
            await message.channel.send("❌ You must be in a voice channel to enable TTS.")
            return
        if not message.author.voice.self_mute:
            await message.channel.send("❌ Please mute your microphone before enabling TTS.")
            return
        
        tts_states = load_json_file(TTS_STATE_FILE)
        tts_states[user_id] = True
        save_json_file(TTS_STATE_FILE, tts_states)
        
        user_settings = load_json_file(USER_SETTINGS_FILE)
        engine = user_settings.get(user_id, {}).get('tts_engine', DEFAULT_TTS_ENGINE)
        
        await message.channel.send(f"⚡ **Lightning TTS enabled** using **{engine}** engine!")

    async def handle_disable_tts(self, message, user_id):
        """Handle disable TTS command"""
        tts_states = load_json_file(TTS_STATE_FILE)
        tts_states.pop(user_id, None)
        save_json_file(TTS_STATE_FILE, tts_states)
        
        await message.channel.send("🔇 **TTS disabled**")

    async def handle_status_command(self, message, user_id):
        """Handle status command"""
        user_settings = load_json_file(USER_SETTINGS_FILE)
        current_engine = user_settings.get(user_id, {}).get('tts_engine', DEFAULT_TTS_ENGINE)
        
        elevenlabs_status = "✅ Ready" if elevenlabs_engine.client else "❌ Not configured"
        coqui_status = "⚡ Lightning-Fast" if coqui_engine.tts else "❌ Not available"
        
        voice_info = await voice_manager.get_user_voice_info(user_id)
        voice_status = "⚡ Lightning-Optimized" if voice_info['coqui_lightning'] else "📝 Standard"
        
        queue_status = await queue_manager.get_status()
        queue_info = f"📊 Queue: {queue_status['queue_length']} requests"
        
        status_embed = discord.Embed(title="⚡ Lightning TTS Status", color=0x00ff00)
        status_embed.add_field(
            name="🎛️ Engine Status",
            value=f"Current: **{current_engine}**\n"
                  f"ElevenLabs: {elevenlabs_status}\n"
                  f"Coqui TTS: {coqui_status}",
            inline=True
        )
        status_embed.add_field(
            name="🎙️ Voice Status",
            value=f"Your Voice: {voice_status}\n{queue_info}",
            inline=True
        )
        
        await message.channel.send(embed=status_embed)

    async def handle_performance_command(self, message):
        """Handle performance metrics command"""
        queue_status = await queue_manager.get_status()
        
        perf_embed = discord.Embed(
            title="⚡ Lightning Performance Metrics",
            color=0x00ff00
        )
        
        perf_embed.add_field(
            name="📊 Queue Performance",
            value=f"Active requests: {queue_status['queue_length']}\n"
                  f"Processing: {'Yes' if queue_status['processing'] else 'No'}\n"
                  f"Estimated wait: {queue_status['estimated_wait_time']}s",
            inline=True
        )
        
        # GPU info if available
        if coqui_engine.tts:
            import torch
            if torch.cuda.is_available():
                gpu_util = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100
                perf_embed.add_field(
                    name="🚀 GPU Performance",
                    value=f"GPU: {torch.cuda.get_device_name(0)}\n"
                          f"Utilization: {gpu_util:.1f}%\n"
                          f"Status: LIGHTNING-FAST",
                    inline=True
                )
        
        await message.channel.send(embed=perf_embed)

    async def handle_queue_command(self, message):
        """Handle queue status command"""
        queue_info = await queue_manager.get_info()
        await message.channel.send(queue_info)

    async def handle_voice_upload(self, message, content, user_id):
        """Handle voice upload commands with proper error handling"""
        is_replacement = content.startswith("!replacevoice")
        
        parts = content.split(" ", 1)
        if len(parts) < 2 or not message.attachments:
            await message.channel.send("❌ Usage: `!uploadvoice <name>` with audio file")
            return
        
        name = parts[1].strip()
        file = message.attachments[0]
        
        processing_msg = await message.channel.send("⚡ Processing voice with lightning speed...")
        
        try:
            # Upload file
            audio_file = await voice_manager.upload_voice(user_id, file, name)
            
            user_settings = load_json_file(USER_SETTINGS_FILE)
            engine = user_settings.get(user_id, {}).get('tts_engine', DEFAULT_TTS_ENGINE)
            
            if is_replacement:
                success, backup_name = await voice_manager.replace_voice(user_id, audio_file, name, engine)
                if success:
                    await processing_msg.edit(content=f"✅ Voice **{name}** replaced! Backup: `{backup_name}`")
            else:
                if engine == "elevenlabs" and elevenlabs_engine.client:
                    await elevenlabs_engine.upload_voice(user_id, audio_file, name)
                    await processing_msg.edit(content=f"✅ Voice **{name}** uploaded to ElevenLabs!")
                elif engine == "coqui" and coqui_engine.tts:
                    await processing_msg.edit(content="⚡ Processing voice in background...")
                    
                    # Use asyncio task instead of threading
                    asyncio.create_task(process_voice_background(user_id, audio_file, name, message.channel))
                    
                else:
                    await processing_msg.edit(content=f"❌ Engine **{engine}** not available")
        
        except Exception as e:
            await processing_msg.edit(content=f"❌ Error: {e}")

    async def handle_my_voice(self, message, user_id):
        """Handle my voice command"""
        try:
            user_settings = load_json_file(USER_SETTINGS_FILE)
            engine = user_settings.get(user_id, {}).get('tts_engine', DEFAULT_TTS_ENGINE)
            
            voice_info = await voice_manager.get_user_voice_info(user_id)
            
            if engine == "elevenlabs":
                if voice_info['elevenlabs_voice']:
                    await message.channel.send("✅ **Custom ElevenLabs voice activated**")
                else:
                    await message.channel.send("❌ No custom ElevenLabs voice found.\n💡 Use `!uploadvoice <name>` to upload one.")
            
            elif engine == "coqui":
                if voice_info['coqui_lightning']:
                    await message.channel.send("✅ **Custom voice activated** (⚡ Lightning-Optimized) for **coqui** engine.")
                elif voice_info['coqui_voice']:
                    await message.channel.send("✅ **Custom voice activated** (Standard) for **coqui** engine.")
                else:
                    await message.channel.send("❌ No custom voice found for **coqui** engine.\n💡 Use `!uploadvoice <name>` to upload one.")
            
        except Exception as e:
            await message.channel.send(f"❌ Error checking voice: {e}")

    async def handle_list_backups(self, message, user_id):
        """Handle list backups command"""
        try:
            backups = await voice_manager.backup_manager.list_voice_backups(user_id)
            
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

    async def handle_restore_backup(self, message, content, user_id):
        """Handle restore backup command"""
        parts = content.split(" ", 1)
        if len(parts) < 2:
            await message.channel.send("❌ Usage: `!restore <backup_name>`")
            return
        
        backup_name = parts[1].strip()
        
        try:
            success, result_msg = await voice_manager.backup_manager.restore_voice_backup(user_id, backup_name)
            
            if success:
                await message.channel.send(f"✅ {result_msg}")
            else:
                await message.channel.send(f"❌ {result_msg}")
                
        except Exception as e:
            await message.channel.send(f"❌ Restore error: {e}")

    async def handle_clear_my_cache(self, message, user_id):
        """Handle clear my cache command"""
        try:
            clean_corrupted_cache(user_id)
            await message.channel.send("✅ Your voice cache has been cleared. Please re-upload your voice if needed.")
        except Exception as e:
            await message.channel.send(f"❌ Error clearing cache: {e}")

    async def handle_clear_cache(self, message, user_id):
        """Handle clear cache command (admin only)"""
        if not check_admin_permissions(message):
            await message.channel.send("❌ Admin permissions required.")
            return
        
        try:
            # Clear all caches
            import shutil
            from .utils import VOICE_CACHE_DIR, PHRASE_CACHE_DIR
            
            for cache_dir in [VOICE_CACHE_DIR, PHRASE_CACHE_DIR]:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir()
            
            await message.channel.send("✅ All caches cleared successfully.")
        except Exception as e:
            await message.channel.send(f"❌ Error clearing caches: {e}")

    async def handle_list_voices(self, message):
        """Handle list voices command"""
        if not elevenlabs_engine.client:
            await message.channel.send("❌ ElevenLabs not configured.")
            return
        
        if not elevenlabs_engine.voices:
            await elevenlabs_engine.fetch_voices()
        
        if elevenlabs_engine.voices:
            voices_list = "\n".join(f"🎤 `{v}`" for v in elevenlabs_engine.voices)
            await message.channel.send(f"🎙️ **Available ElevenLabs Voices:**\n{voices_list}")
        else:
            await message.channel.send("❌ No voices available.")

    async def handle_set_voice(self, message, content, user_id):
        """Handle set voice command"""
        if not elevenlabs_engine.client:
            await message.channel.send("❌ ElevenLabs not configured.")
            return
        
        name = content[6:].strip()
        if name in elevenlabs_engine.voices:
            elevenlabs_engine.current_voice_id = elevenlabs_engine.voices[name]
            await message.channel.send(f"✅ ElevenLabs voice changed to **{name}**")
        else:
            await message.channel.send("❌ Voice not found. Use `!voices` to see available voices.")

    async def handle_stats_command(self, message, user_id):
        """Handle stats command"""
        analytics = load_json_file(ANALYTICS_FILE)
        
        if user_id in analytics.get("usage", {}):
            user_data = analytics["usage"][user_id]
            last_used = datetime.fromtimestamp(user_data["last_used"]).strftime("%Y-%m-%d %H:%M")
            
            stats_msg = f"📊 **Your TTS Statistics**\n\n"
            stats_msg += f"🎤 Total requests: {user_data['count']}\n"
            stats_msg += f"📅 Last used: {last_used}\n"
            
            await message.channel.send(stats_msg)
        else:
            await message.channel.send("📊 No usage statistics found. Start using TTS to see your stats!")

    async def handle_popular_command(self, message):
        """Handle popular phrases command"""
        analytics = load_json_file(ANALYTICS_FILE)
        popular_phrases = sorted(analytics.get("popular_phrases", {}).items(), key=lambda x: x[1], reverse=True)[:10]
        
        if not popular_phrases:
            await message.channel.send("📈 No popular phrases data available yet.")
            return
        
        popular_msg = "🔥 **Most Popular Phrases:**\n\n"
        for i, (phrase, count) in enumerate(popular_phrases, 1):
            popular_msg += f"{i}. \"{phrase[:50]}{'...' if len(phrase) > 50 else ''}\" - {count} uses\n"
        
        await message.channel.send(popular_msg)

    async def handle_blacklist_command(self, message, content):
        """Handle blacklist command (admin only)"""
        parts = content.split()
        if len(parts) == 2:
            blacklist = load_json_file(BLACKLIST_FILE)
            blacklist[parts[1]] = True
            save_json_file(BLACKLIST_FILE, blacklist)
            await message.channel.send(f"⛔ **User blacklisted:** `{parts[1]}`")

    async def handle_disconnect(self, message):
        """Handle disconnect command"""
        vc = discord.utils.get(self.voice_clients, guild=message.guild)
        if vc:
            await vc.disconnect()
            await message.channel.send("✅ **Bot disconnected**")
        else:
            await message.channel.send("❌ Not connected to any voice channel")

# Initialize bot
bot = LightningTTSBot()

async def main():
    """Main function to run the bot"""
    try:
        await bot.start(DISCORD_TOKEN)
    except Exception as e:
        print(f"❌ Bot startup failed: {e}")
        raise
