# 🎙️ Discord TTS Bot (ElevenLabs)

A powerful Discord bot that lets users convert text to speech using the ElevenLabs API. Includes support for personal voice cloning, voice channel playback, user-specific voice selection, and more.

---

## ⚙️ Features

- 🔊 Text-to-speech using ElevenLabs voices
- 🧠 Clone your own voice using a voice sample
- 🎭 Use multiple ElevenLabs voices and switch dynamically
- 🎤 Mute-check: Users must mute mic to use TTS
- 🔒 Admin-only bypass for testing
- ⛔ Blacklist users from using TTS
- 📥 Upload `.mp3` or `.wav` files to train voices
- ⏳ TTS queue when multiple users send text simultaneously

---

## 🛠️ Installation

```bash
git clone https://github.com/C0deRatoR/discord-tts-bot.git
cd discord-tts-bot
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux

pip install -r requirements.txt
