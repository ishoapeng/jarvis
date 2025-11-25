"""
Orchestrator Layer
Coordinates between LLM, Memory, and System Actions
"""

from typing import Dict, Optional
import json
import re
from memory_layer import MemoryLayer
from system_actions import SystemActions
from vllm import SamplingParams


class ActionOutputParser:
    """Parse LLM output to extract actions."""
    
    def parse(self, text: str) -> Dict:
        """Parse LLM response to extract action and parameters."""
        # Look for JSON action format
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                action_data = json.loads(json_match.group())
                return action_data
            except:
                pass
        
        # Look for action: format
        action_match = re.search(r'action:\s*(\w+)', text, re.IGNORECASE)
        if action_match:
            return {
                "action": action_match.group(1),
                "response": text
            }
        
        # Default: just return the text as response
        return {
            "action": None,
            "response": text
        }


class Orchestrator:
    """Orchestrates interactions between LLM, Memory, and System Actions."""
    
    def __init__(self, llm, memory_layer: MemoryLayer, system_actions: SystemActions):
        """Initialize orchestrator."""
        self.llm = llm
        self.memory = memory_layer
        self.actions = system_actions
        self.output_parser = ActionOutputParser()
        
        # Sampling parameters for vLLM
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256
        )
    
    def process_query(self, user_input: str) -> Dict:
        """Process a user query through the full pipeline."""
        if not self.llm:
            return {
                "response": "LLM is not available",
                "action": None,
                "action_result": None
            }
        
        # Get context from memory
        context = self.memory.get_context_for_llm(user_input)
        
        # Get available actions
        available_actions = ", ".join(self.actions.actions.keys())
        
        # Create prompt
        prompt = f"""You are Jarvis, a helpful AI assistant. You have access to system actions and conversation history.

Available system actions: {available_actions}

Previous conversation context:
{context}

User: {user_input}

Assistant: Respond naturally. If you need to perform a system action, respond in JSON format:
{{"action": "action_name", "parameters": {{"param": "value"}}, "response": "what to say to user"}}

If no action is needed, just respond normally.
"""
        
        # Generate response using vLLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        # Parse response
        parsed = self.output_parser.parse(response_text)
        
        # Execute action if needed
        action_result = None
        if parsed.get("action"):
            action_result = self.actions.execute_action(
                parsed["action"],
                **parsed.get("parameters", {})
            )
        
        # Determine final response
        if parsed.get("response"):
            final_response = parsed["response"]
        elif action_result and action_result.get("message"):
            final_response = action_result["message"]
        else:
            final_response = response_text
        
        # Store in memory
        self.memory.add_conversation(user_input, final_response)
        
        return {
            "response": final_response,
            "action": parsed.get("action"),
            "action_result": action_result
        }

