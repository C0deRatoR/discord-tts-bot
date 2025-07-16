"""
Queue System Module for Lightning Discord TTS Bot
Handles TTS request queuing and prevents audio conflicts
"""

import asyncio
import time
from collections import deque
from datetime import datetime

from .utils import (
    load_json_file,
    save_json_file,
    ANALYTICS_FILE,
    user_stats,
    check_admin_permissions
)

class TTSQueue:
    """Smart TTS queue system with priority support and bot instance handling"""
    
    def __init__(self):
        self.queue = deque()
        self.processing = False
        self.current_user = None
        self.queue_lock = asyncio.Lock()
        self.processing_task = None
        self.bot_instance = None
        
    def set_bot_instance(self, bot):
        """Set the bot instance for voice operations"""
        self.bot_instance = bot
        
    async def start_queue_processor(self):
        """Start the queue processing task"""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_queue())
            print("🔄 TTS Queue system started")
    
    async def stop_queue_processor(self):
        """Stop the queue processing task"""
        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None
            print("🛑 TTS Queue system stopped")
    
    async def add_request(self, user_id, text, voice_channel, engine, message_author, priority=0):
        """Add TTS request to queue with priority support"""
        async with self.queue_lock:
            request = {
                'user_id': user_id,
                'text': text,
                'voice_channel': voice_channel,
                'engine': engine,
                'message_author': message_author,
                'priority': priority,
                'timestamp': time.time(),
                'request_id': f"{user_id}_{int(time.time())}"
            }
            
            # Admin requests get priority
            if priority > 0:
                self.queue.appendleft(request)
                position = 1
            else:
                self.queue.append(request)
                position = len(self.queue)
            
            print(f"📥 Added to queue: {text[:30]}... (Position: {position})")
            return position
    
    async def _process_queue(self):
        """Process TTS queue continuously with error handling"""
        while True:
            try:
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
                
                # Process the request with error handling
                await self._execute_tts_request(request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Queue processing error: {e}")
            finally:
                self.processing = False
                self.current_user = None
                await asyncio.sleep(0.2)
    
    async def _execute_tts_request(self, request):
        """Execute a TTS request with proper bot instance handling"""
        try:
            from .tts_engine import generate_tts
            import discord
            
            print(f"🎤 Processing: {request['text'][:30]}... for user {request['user_id']}")
            
            # Generate TTS
            start_time = time.time()
            try:
                audio_file = await generate_tts(request['text'], request['user_id'], request['engine'])
            except Exception as e:
                print(f"❌ TTS generation failed: {e}")
                return
            
            generation_time = time.time() - start_time
            
            # Use the stored bot instance
            if not self.bot_instance:
                print("❌ Bot instance not available")
                return
            
            voice_channel = request['voice_channel']
            guild = voice_channel.guild
            
            try:
                vc = discord.utils.get(self.bot_instance.voice_clients, guild=guild)
                
                if not vc:
                    vc = await voice_channel.connect()
                elif vc.channel != voice_channel:
                    await vc.move_to(voice_channel)
                
                # Play audio
                def after_play(error):
                    if error:
                        print(f"❌ Audio playback error: {error}")
                    
                    # Clean up temporary files
                    try:
                        import os
                        if not audio_file.startswith("lightning_tts_"):
                            os.remove(audio_file)
                    except:
                        pass
                
                source = discord.FFmpegPCMAudio(audio_file)
                vc.play(source, after=after_play)
                
                # Update analytics with error handling
                try:
                    await self._update_analytics(request['user_id'], request['text'], generation_time)
                except Exception as e:
                    print(f"❌ Analytics update error: {e}")
                
                print(f"✅ Completed: {request['text'][:30]}... ({generation_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ Voice connection error: {e}")
                
        except Exception as e:
            print(f"❌ TTS execution error: {e}")
    
    async def _update_analytics(self, user_id, text, generation_time):
        """Update usage analytics with proper error handling"""
        try:
            # Load analytics with proper default structure
            default_analytics = {
                "usage": {},
                "popular_phrases": {},
                "voice_uploads": {},
                "queue_stats": {
                    "total_requests": 0,
                    "avg_generation_time": 0.0,
                    "requests_by_hour": {}
                }
            }
            
            analytics = load_json_file(ANALYTICS_FILE, default_analytics)
            
            # Ensure all required keys exist
            if "usage" not in analytics:
                analytics["usage"] = {}
            if "popular_phrases" not in analytics:
                analytics["popular_phrases"] = {}
            if "voice_uploads" not in analytics:
                analytics["voice_uploads"] = {}
            if "queue_stats" not in analytics:
                analytics["queue_stats"] = {
                    "total_requests": 0,
                    "avg_generation_time": 0.0,
                    "requests_by_hour": {}
                }
            
            # Update user usage
            if user_id not in analytics["usage"]:
                analytics["usage"][user_id] = {"count": 0, "last_used": time.time()}
            
            analytics["usage"][user_id]["count"] += 1
            analytics["usage"][user_id]["last_used"] = time.time()
            
            # Update popular phrases
            clean_text = text.lower().strip()
            if clean_text in analytics["popular_phrases"]:
                analytics["popular_phrases"][clean_text] += 1
            else:
                analytics["popular_phrases"][clean_text] = 1
            
            # Update queue stats
            queue_stats = analytics["queue_stats"]
            queue_stats["total_requests"] += 1
            
            # Update average generation time
            total_requests = queue_stats["total_requests"]
            current_avg = queue_stats.get("avg_generation_time", 0.0)
            queue_stats["avg_generation_time"] = (
                (current_avg * (total_requests - 1) + generation_time) / total_requests
            )
            
            # Update hourly stats
            hour = datetime.now().strftime("%Y-%m-%d %H")
            if "requests_by_hour" not in queue_stats:
                queue_stats["requests_by_hour"] = {}
            
            if hour not in queue_stats["requests_by_hour"]:
                queue_stats["requests_by_hour"][hour] = 0
            queue_stats["requests_by_hour"][hour] += 1
            
            # Save analytics
            save_json_file(ANALYTICS_FILE, analytics)
            
            # Update user stats
            user_stats[user_id]["phrases"].append(clean_text)
            user_stats[user_id]["last_used"] = time.time()
            
        except Exception as e:
            print(f"❌ Analytics update failed: {e}")
    
    def get_queue_status(self):
        """Get current queue status"""
        return {
            'queue_length': len(self.queue),
            'processing': self.processing,
            'current_user': self.current_user,
            'estimated_wait_time': len(self.queue) * 2
        }
    
    async def clear_queue(self):
        """Clear all queued requests"""
        async with self.queue_lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            return cleared_count
    
    async def remove_user_requests(self, user_id):
        """Remove all requests from a specific user"""
        async with self.queue_lock:
            original_length = len(self.queue)
            self.queue = deque([req for req in self.queue if req['user_id'] != user_id])
            removed_count = original_length - len(self.queue)
            return removed_count
    
    async def get_queue_info(self):
        """Get detailed queue information"""
        async with self.queue_lock:
            if not self.queue:
                return "✅ Queue is empty - ready for instant TTS!"
            
            info = f"📊 **TTS Queue Status**\n"
            info += f"🔄 Requests in queue: {len(self.queue)}\n"
            
            if self.processing:
                info += f"⚡ Currently processing: <@{self.current_user}>\n"
            
            info += f"⏱️ Estimated wait: {len(self.queue) * 2}s\n"
            
            # Show next few requests
            if len(self.queue) > 0:
                info += "\n📋 **Next in queue:**\n"
                for i, request in enumerate(list(self.queue)[:3], 1):
                    text_preview = request['text'][:20] + "..." if len(request['text']) > 20 else request['text']
                    info += f"{i}. <@{request['user_id']}>: \"{text_preview}\"\n"
                
                if len(self.queue) > 3:
                    info += f"... and {len(self.queue) - 3} more requests\n"
            
            return info

class QueueManager:
    """Main queue management system"""
    
    def __init__(self):
        self.tts_queue = TTSQueue()
        self.active = False
    
    def set_bot_instance(self, bot):
        """Set the bot instance for voice operations"""
        self.tts_queue.set_bot_instance(bot)
    
    async def start(self):
        """Start the queue system"""
        if not self.active:
            await self.tts_queue.start_queue_processor()
            self.active = True
    
    async def stop(self):
        """Stop the queue system"""
        if self.active:
            await self.tts_queue.stop_queue_processor()
            self.active = False
    
    async def add_tts_request(self, user_id, text, voice_channel, engine, message_author):
        """Add TTS request with priority handling"""
        priority = 1 if check_admin_permissions(message_author) else 0
        
        position = await self.tts_queue.add_request(
            user_id=user_id,
            text=text,
            voice_channel=voice_channel,
            engine=engine,
            message_author=message_author,
            priority=priority
        )
        
        return position
    
    async def get_status(self):
        """Get queue status"""
        return self.tts_queue.get_queue_status()
    
    async def get_info(self):
        """Get detailed queue information"""
        return await self.tts_queue.get_queue_info()
    
    async def clear_queue(self):
        """Clear all queued requests"""
        return await self.tts_queue.clear_queue()
    
    async def remove_user_requests(self, user_id):
        """Remove all requests from a specific user"""
        return await self.tts_queue.remove_user_requests(user_id)

# Initialize queue manager
queue_manager = QueueManager()
