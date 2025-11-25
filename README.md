# Jarvis Voice Assistant

A local, privacy-focused voice assistant with wake word detection, powered by vLLM and Whisper.

## Architecture

```
┌──────────────────────────┐
│   Voice Interface        │  ←→ Whisper (ASR) / Piper (TTS)
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   LLM Core (vLLM)       │  ←→ Qwen2.5-3B-Instruct
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Orchestrator Layer     │  ←→ Coordinates all components
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Memory / Vector DB     │  ←→ ChromaDB for conversation history
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   System Actions API     │  ←→ OS, Browser, Apps, etc.
└──────────────────────────┘
```

## Features

- **Wake Word Detection**: Listens for "Hey Jarvis" to activate
- **Voice Recognition**: Uses OpenAI Whisper for accurate speech-to-text
- **Text-to-Speech**: Uses Piper TTS for natural voice responses
- **Local LLM**: Runs Qwen2.5-3B-Instruct locally via vLLM (no cloud needed)
- **Memory**: Maintains conversation context using ChromaDB
- **System Actions**: Can open apps, check time, run commands, etc.

## Requirements

- Python 3.13+
- NVIDIA GPU with 8GB+ VRAM (RTX 3070Ti or better)
- Microphone
- Audio output (speakers/headphones)

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Install system dependencies (Arch Linux):
```bash
# For audio
sudo pacman -S portaudio python-pyaudio

# For Piper TTS (optional, falls back to espeak)
# Download from: https://github.com/rhasspy/piper/releases
```

## Usage

Start Jarvis:
```bash
./start.sh
```

Or directly:
```bash
uv run python jarvis.py
```

## Commands

- **"Hey Jarvis"** - Wake word to activate
- **"Open Cursor"** - Opens Cursor editor
- **"Open Browser"** - Opens web browser
- **"What time is it?"** - Gets current time
- **"What's the date?"** - Gets current date
- Any other natural language query - Handled by the LLM

## Configuration

### Memory Layer
Conversation history is stored in `./chroma_db/` directory.

### System Actions
Add custom actions in `system_actions.py` by extending the `SystemActions` class.

### Voice Settings
Adjust wake word, TTS voice, and audio settings in `voice_interface.py`.

## Architecture Details

### Voice Interface (`voice_interface.py`)
- **ASR**: OpenAI Whisper (base model for speed, can use 'small' for accuracy)
- **TTS**: Piper TTS (falls back to espeak/festival if unavailable)
- Handles wake word detection and command listening

### LLM Core (`jarvis.py`)
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Engine**: vLLM for efficient inference
- Optimized for 8GB VRAM

### Orchestrator (`orchestrator.py`)
- Coordinates between LLM, Memory, and System Actions
- Parses LLM responses to extract actions
- Manages conversation flow

### Memory Layer (`memory_layer.py`)
- **Storage**: ChromaDB (vector database)
- Stores conversation history
- Provides context for LLM prompts
- Enables semantic search of past conversations

### System Actions (`system_actions.py`)
- Executes system-level commands
- Opens applications
- Retrieves system information
- Extensible for custom actions

## Troubleshooting

### Audio Issues
- Ensure microphone is connected and working
- Check audio permissions
- Try: `arecord -l` to list audio devices

### GPU Memory Issues
- Reduce `gpu_memory_utilization` in `jarvis.py`
- Reduce `max_model_len` if needed
- Close other GPU-intensive applications

### Whisper Model Download
- First run will download Whisper model (~150MB)
- Ensure internet connection for initial setup

## License

MIT

# jarvis
