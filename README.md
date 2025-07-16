# ⚡ Lightning Discord TTS Bot

A high-performance Discord Text-to-Speech bot with dual engine support, GPU acceleration, and advanced voice management features.

## 🚀 Features

### **⚡ Lightning Performance**
- GPU-accelerated processing with CUDA support
- Quantized models for 2x speed improvement
- Phrase caching for instant responses
- Parallel processing with CUDA streams

### **🎙️ Dual TTS Engine Support**
- **Coqui TTS**: Free, high-quality voice synthesis with cloning
- **ElevenLabs**: Premium quality with professional voices
- Seamless switching between engines per user

### **🔄 Advanced Voice Management**
- Custom voice cloning from audio samples
- Voice replacement with automatic backups
- Voice restore system with version history
- Cross-engine voice compatibility

### **📊 Smart Queue System**
- No audio conflicts when multiple users speak
- Priority system for administrators
- Real-time queue status and monitoring
- Automatic queue processing

### **📈 Analytics & Insights**
- Usage statistics and tracking
- Popular phrases analysis
- Performance metrics
- Voice upload history

## 🛠️ Installation

### **Prerequisites**
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- Discord Bot Token
- ElevenLabs API Key (optional)

### **Setup Steps**

1. **Clone the repository:**
```
git clone https://github.com/C0deRatoR/discord-tts-bot.git
cd discord-tts-bot
```

2. **Create conda environment:**
```
conda create -n tts_bot python=3.10
conda activate tts_bot
```

3. **Install dependencies:**
```
# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

4. **Setup environment variables:**
Create a `.env` file in the project root:
```
DISCORD_TOKEN=your_discord_bot_token_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

5. **Run the bot:**
```
python run.py
```

## 📁 Project Structure

```
discord-tts-bot/
├── run.py                  # Main entry point
├── bot/                    # Bot modules
│   ├── __init__.py
│   ├── main.py             # Core bot logic
│   ├── tts_engine.py       # TTS functionality
│   ├── voice_manager.py    # Voice management
│   ├── queue_system.py     # Queue processing
│   └── utils.py            # Utility functions
├── data/                   # Data storage
│   ├── uploads/            # Voice uploads
│   ├── voice_cache/        # Processed voices
│   ├── phrase_cache/       # Cached phrases
│   └── voice_backups/      # Voice backups
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── LICENSE                # MIT License
```

## 🎮 Commands

### **🎤 Basic TTS**
| Command | Description |
|---------|-------------|
| `!tts ` | Lightning-fast speech synthesis |
| `!enabletts` | Enable auto-TTS (mute mic first) |
| `!disabletts` | Disable auto-TTS |

### **🔧 Engine Control**
| Command | Description |
|---------|-------------|
| `!engine ` | Switch TTS engine |
| `!status` | Check system status |
| `!performance` | Performance metrics |
| `!queue` | Check TTS queue status |

### **🎙️ Voice Management**
| Command | Description |
|---------|-------------|
| `!uploadvoice ` | Upload new voice |
| `!replacevoice ` | Replace existing voice |
| `!myvoice` | Use your custom voice |
| `!voices` | List ElevenLabs voices |

### **🔄 Voice Backup System**
| Command | Description |
|---------|-------------|
| `!backups` | List your voice backups |
| `!restore ` | Restore voice backup |
| `!voicehistory` | View voice upload history |

### **📊 Analytics**
| Command | Description |
|---------|-------------|
| `!stats` | Your usage statistics |
| `!popular` | Most popular phrases |
| `!analytics` | Server analytics (admin only) |

## 🚀 Performance

- **Cold Start**: 2-3 seconds (first generation)
- **Warmed Up**: 0.5-1 second per request
- **Cached Phrases**: < 0.1 seconds (instant)
- **Queue Processing**: Multiple users handled seamlessly

## 🔧 Configuration

The bot automatically configures itself based on available hardware:
- **GPU Detection**: Automatic CUDA support
- **Memory Optimization**: 95% GPU memory allocation
- **Queue Management**: Smart request processing
- **Content Filtering**: Automatic spam detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Coqui TTS** - Open-source TTS engine
- **ElevenLabs** - Premium TTS API
- **Discord.py** - Discord API wrapper
- **PyTorch** - Deep learning framework

**Made with ⚡ by C0deRatoR**