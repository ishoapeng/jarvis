"""
Memory / Vector DB Layer
Handles conversation memory and vector storage using Chroma
"""

import chromadb
from chromadb.config import Settings
import os
from datetime import datetime
from typing import List, Dict, Optional
import json


class MemoryLayer:
    """Manages conversation memory and vector storage."""
    
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize memory layer with ChromaDB."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collections
        self.conversation_collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.memory_collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_conversation(self, user_input: str, assistant_response: str, metadata: Optional[Dict] = None):
        """Add a conversation turn to memory."""
        timestamp = datetime.now().isoformat()
        conversation_text = f"User: {user_input}\nAssistant: {assistant_response}"
        
        # Generate ID
        conversation_id = f"conv_{datetime.now().timestamp()}"
        
        # Store in Chroma
        self.conversation_collection.add(
            ids=[conversation_id],
            documents=[conversation_text],
            metadatas=[{
                "timestamp": timestamp,
                "user_input": user_input,
                "assistant_response": assistant_response,
                **(metadata or {})
            }]
        )
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Get recent conversations for context."""
        results = self.conversation_collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        conversations = []
        for i, metadata in enumerate(results["metadatas"]):
            conversations.append({
                "user_input": metadata.get("user_input", ""),
                "assistant_response": metadata.get("assistant_response", ""),
                "timestamp": metadata.get("timestamp", "")
            })
        
        return conversations
    
    def search_similar_conversations(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for similar past conversations."""
        results = self.conversation_collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        similar = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                similar.append({
                    "user_input": metadata.get("user_input", ""),
                    "assistant_response": metadata.get("assistant_response", ""),
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 0
                })
        
        return similar
    
    def add_memory(self, memory_text: str, category: str = "general"):
        """Add a persistent memory."""
        memory_id = f"mem_{datetime.now().timestamp()}"
        self.memory_collection.add(
            ids=[memory_id],
            documents=[memory_text],
            metadatas={
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_context_for_llm(self, current_query: str, max_conversations: int = 3) -> str:
        """Get relevant context for LLM prompt."""
        # Get recent conversations
        recent = self.get_recent_conversations(limit=max_conversations)
        
        # Search for similar conversations
        similar = self.search_similar_conversations(current_query, limit=2)
        
        context = "Previous conversation context:\n"
        for conv in recent:
            context += f"User: {conv['user_input']}\nAssistant: {conv['assistant_response']}\n\n"
        
        if similar:
            context += "Similar past conversations:\n"
            for sim in similar:
                context += f"User: {sim['user_input']}\nAssistant: {sim['assistant_response']}\n\n"
        
        return context

