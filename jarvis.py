#!/usr/bin/env python3
"""
Jarvis Voice Assistant - Main Entry Point
Architecture:
    Voice Interface (Whisper/Piper) → LLM Core (vLLM) → Orchestrator (LangChain) 
    → Memory/Vector DB (Chroma) → System Actions API
"""

import os
import sys
import threading
from vllm import LLM, SamplingParams

# Import all layers
from voice_interface import VoiceInterface
from memory_layer import MemoryLayer
from system_actions import SystemActions
from orchestrator import Orchestrator

# Set PyTorch memory allocation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


class JarvisAssistant:
    """Main Jarvis assistant coordinating all layers."""
    
    def __init__(self):
        """Initialize all layers of the Jarvis architecture."""
        print("=" * 60)
        print("Initializing Jarvis Assistant")
        print("=" * 60)
        
        # Initialize Voice Interface (Whisper/Piper)
        print("\n[1/5] Initializing Voice Interface...")
        self.voice = VoiceInterface(wake_word="hey jarvis")
        
        # Initialize Memory Layer (Chroma)
        print("\n[2/5] Initializing Memory Layer...")
        self.memory = MemoryLayer()
        
        # Initialize System Actions
        print("\n[3/5] Initializing System Actions...")
        self.actions = SystemActions()
        
        # Initialize LLM Core (vLLM)
        print("\n[4/5] Initializing LLM Core (vLLM)...")
        self.llm = self._init_llm()
        
        # Initialize Orchestrator (LangChain)
        print("\n[5/5] Initializing Orchestrator...")
        self.orchestrator = Orchestrator(
            llm=self.llm,
            memory_layer=self.memory,
            system_actions=self.actions
        )
        
        print("\n" + "=" * 60)
        print("Jarvis is ready!")
        print("=" * 60)
    
    def _init_llm(self):
        """Initialize the vLLM model."""
        try:
            llm = LLM(
                model="Qwen/Qwen2.5-3B-Instruct",
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1,
                trust_remote_code=True,
                dtype="bfloat16",
            )
            print("✓ LLM Core loaded successfully")
            return llm
        except Exception as e:
            print(f"✗ Error loading LLM: {e}")
            print("Continuing without LLM (limited functionality)...")
            return None
    
    def handle_wake_word_detected(self):
        """Called when wake word is detected."""
        self.voice.speak("Yes, I'm listening")
        
        # Listen for command
        command = self.voice.listen_for_command(duration=5)
        
        if not command:
            self.voice.speak("I didn't hear anything")
            return
        
        print(f"\nProcessing command: {command}")
        
        # Process through orchestrator
        if self.orchestrator:
            try:
                result = self.orchestrator.process_query(command)
                response = result.get("response", "I'm not sure how to help with that")
                
                # Speak the response
                self.voice.speak(response)
                
                # Log action if executed
                if result.get("action"):
                    print(f"Executed action: {result['action']}")
                    if result.get("action_result"):
                        print(f"Action result: {result['action_result']}")
                        
            except Exception as e:
                print(f"Error processing query: {e}")
                self.voice.speak("Sorry, I encountered an error")
        else:
            # Fallback: simple command matching
            self._handle_simple_command(command)
    
    def _handle_simple_command(self, command: str):
        """Simple command handler fallback when LLM is not available."""
        command_lower = command.lower()
        
        if "open cursor" in command_lower:
            result = self.actions.open_cursor()
            self.voice.speak(result.get("message", "Opening Cursor"))
        elif "open browser" in command_lower:
            result = self.actions.open_browser()
            self.voice.speak(result.get("message", "Opening browser"))
        elif "what time" in command_lower:
            result = self.actions.get_time()
            self.voice.speak(result.get("message", "I don't know the time"))
        else:
            self.voice.speak("I'm not sure how to help with that")
    
    def run(self):
        """Start the Jarvis assistant."""
        try:
            # Greet user
            self.voice.speak("Jarvis is online and ready")
            
            # Start listening for wake word
            self.voice.listen_for_wake_word(self.handle_wake_word_detected)
            
        except KeyboardInterrupt:
            print("\n\nShutting down Jarvis...")
            self.voice.speak("Goodbye")
            self.voice.cleanup()
            sys.exit(0)
        except Exception as e:
            print(f"Fatal error: {e}")
            self.voice.cleanup()
            sys.exit(1)


def main():
    """Main entry point."""
    try:
        assistant = JarvisAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to start Jarvis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
