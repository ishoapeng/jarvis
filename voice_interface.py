"""
Voice Interface Layer
Handles ASR (Whisper) and TTS (Piper)
"""

import whisper
import pyaudio
import wave
import numpy as np
import threading
import queue
import os
import subprocess
import tempfile


class VoiceInterface:
    """Handles speech recognition (Whisper) and text-to-speech (Piper)."""
    
    def __init__(self, wake_word="hey jarvis"):
        """Initialize voice interface with Whisper and Piper."""
        self.wake_word = wake_word.lower()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")  # Use 'base' for speed, 'small' for accuracy
        print("Whisper model loaded")
        
        # Audio settings
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.audio = pyaudio.PyAudio()
        
        # Check for Piper TTS
        self.piper_available = self._check_piper()
    
    def _check_piper(self):
        """Check if Piper TTS CLI is available with voices."""
        # Check for CLI piper
        try:
            result = subprocess.run(
                ["which", "piper"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                # Try to list voices to see if any are installed
                try:
                    result = subprocess.run(
                        ["piper", "--list-voices"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout and len(result.stdout.strip()) > 0:
                        self.piper_voices = result.stdout.strip().split('\n')
                        return True
                except:
                    pass
        except:
            pass
        
        self.piper_voices = []
        return False
    
    def record_audio(self, duration=5):
        """Record audio from microphone."""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper."""
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
            
            try:
                result = self.whisper_model.transcribe(tmp_path, language="en")
                text = result["text"].strip().lower()
                return text
            finally:
                os.unlink(tmp_path)
    
    def listen_for_wake_word(self, callback):
        """Continuously listen for wake word."""
        print(f"Listening for wake word: '{self.wake_word}'")
        
        while True:
            try:
                # Record short audio chunks
                audio_data = self.record_audio(duration=3)
                text = self.transcribe_audio(audio_data)
                
                if text and self.wake_word in text:
                    print(f"Wake word detected! Heard: {text}")
                    callback()
                else:
                    print(f"Heard: {text}" if text else "Listening...")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in wake word detection: {e}")
    
    def listen_for_command(self, duration=5):
        """Listen for a voice command."""
        print("Listening for command...")
        audio_data = self.record_audio(duration=duration)
        text = self.transcribe_audio(audio_data)
        return text
    
    def speak(self, text):
        """Convert text to speech using Piper or fallback."""
        print(f"Jarvis: {text}")
        
        # Try Piper CLI if available
        if self.piper_available and hasattr(self, 'piper_voices') and self.piper_voices:
            try:
                # Use first available voice
                voice = self.piper_voices[0].split()[0] if self.piper_voices else "en_US-lessac-medium"
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    output_path = tmp_file.name
                
                result = subprocess.run(
                    ["piper", "--model", voice, "--output_file", output_path],
                    input=text.encode(),
                    capture_output=True,
                    timeout=10
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    # Play the audio
                    subprocess.run(["aplay", output_path], check=True, timeout=5)
                    os.unlink(output_path)
                    return
                else:
                    # If specific voice failed, try without specifying (uses default)
                    try:
                        result = subprocess.run(
                            ["piper", "--output_file", output_path],
                            input=text.encode(),
                            capture_output=True,
                            timeout=10
                        )
                        if result.returncode == 0 and os.path.exists(output_path):
                            subprocess.run(["aplay", output_path], check=True, timeout=5)
                            os.unlink(output_path)
                            return
                    except:
                        pass
            except Exception as e:
                print(f"Piper TTS error: {e}")
        
        # Fallback to espeak (more reliable)
        self._fallback_tts(text)
    
    def _fallback_tts(self, text):
        """Fallback TTS using espeak, festival, or spd-say."""
        # Try espeak first (most common on Linux)
        try:
            subprocess.run(
                ["espeak", "-s", "150", text],
                check=True,
                timeout=10
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        
        # Try spd-say (Speech Dispatcher)
        try:
            subprocess.run(
                ["spd-say", text],
                check=True,
                timeout=10
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        
        # Try festival
        try:
            subprocess.run(
                ["festival", "--tts"],
                input=text.encode(),
                check=True,
                timeout=10
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
        
        # Last resort: just print
        print(f"[TTS not available] Would say: {text}")
    
    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()

