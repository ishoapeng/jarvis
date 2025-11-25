"""
System Actions API
Handles OS, Browser, IoT, Email, and other system interactions
"""

import subprocess
import os
import webbrowser
from typing import Dict, Callable, Optional
import json


class SystemActions:
    """Handles system-level actions and integrations."""
    
    def __init__(self):
        """Initialize system actions."""
        self.actions: Dict[str, Callable] = {
            "open_cursor": self.open_cursor,
            "open_browser": self.open_browser,
            "open_terminal": self.open_terminal,
            "get_time": self.get_time,
            "get_date": self.get_date,
            "list_files": self.list_files,
            "run_command": self.run_command,
        }
    
    def open_cursor(self, **kwargs) -> Dict:
        """Open Cursor editor."""
        try:
            subprocess.Popen(
                ["cursor"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return {"success": True, "message": "Cursor opened"}
        except Exception as e:
            return {"success": False, "message": f"Failed to open Cursor: {e}"}
    
    def open_browser(self, url: Optional[str] = None, **kwargs) -> Dict:
        """Open web browser."""
        try:
            if url:
                webbrowser.open(url)
                return {"success": True, "message": f"Opened {url}"}
            else:
                # Try to open default browser
                browsers = ["firefox", "chrome", "chromium", "brave"]
                for browser in browsers:
                    try:
                        subprocess.Popen(
                            [browser],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        return {"success": True, "message": f"Opened {browser}"}
                    except FileNotFoundError:
                        continue
                return {"success": False, "message": "No browser found"}
        except Exception as e:
            return {"success": False, "message": f"Failed to open browser: {e}"}
    
    def open_terminal(self, **kwargs) -> Dict:
        """Open terminal."""
        try:
            terminals = ["gnome-terminal", "konsole", "xterm", "alacritty"]
            for term in terminals:
                try:
                    subprocess.Popen(
                        [term],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return {"success": True, "message": f"Opened {term}"}
                except FileNotFoundError:
                    continue
            return {"success": False, "message": "No terminal found"}
        except Exception as e:
            return {"success": False, "message": f"Failed to open terminal: {e}"}
    
    def get_time(self, **kwargs) -> Dict:
        """Get current time."""
        from datetime import datetime
        current_time = datetime.now().strftime("%I:%M %p")
        return {"success": True, "message": f"The time is {current_time}", "data": current_time}
    
    def get_date(self, **kwargs) -> Dict:
        """Get current date."""
        from datetime import datetime
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return {"success": True, "message": f"Today is {current_date}", "data": current_date}
    
    def list_files(self, directory: Optional[str] = None, **kwargs) -> Dict:
        """List files in directory."""
        try:
            dir_path = directory or os.getcwd()
            files = os.listdir(dir_path)
            return {
                "success": True,
                "message": f"Found {len(files)} items in {dir_path}",
                "data": files[:10]  # Limit to first 10
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to list files: {e}"}
    
    def run_command(self, command: str, **kwargs) -> Dict:
        """Run a system command (with safety checks)."""
        # Safety: only allow certain commands
        allowed_commands = ["ls", "pwd", "date", "whoami"]
        cmd_parts = command.split()
        
        if cmd_parts[0] not in allowed_commands:
            return {
                "success": False,
                "message": f"Command '{cmd_parts[0]}' is not allowed for security reasons"
            }
        
        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=5
            )
            return {
                "success": True,
                "message": result.stdout or "Command executed",
                "data": result.stdout
            }
        except Exception as e:
            return {"success": False, "message": f"Command failed: {e}"}
    
    def execute_action(self, action_name: str, **kwargs) -> Dict:
        """Execute a system action by name."""
        if action_name in self.actions:
            return self.actions[action_name](**kwargs)
        else:
            return {"success": False, "message": f"Unknown action: {action_name}"}

