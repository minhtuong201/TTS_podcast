#!/usr/bin/env python3
"""
Get Voice Information from ElevenLabs API

This script queries the ElevenLabs API to get detailed information
about specific voice IDs including their names and characteristics.
"""

import os
import requests
import json
from typing import Dict, Any, Optional

def get_voice_info(voice_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get voice information from ElevenLabs API
    
    Args:
        voice_id: The voice ID to query
        api_key: ElevenLabs API key (if not provided, looks for ELEVENLABS_API_KEY env var)
    
    Returns:
        Dictionary containing voice information
    """
    if not api_key:
        api_key = os.getenv('ELEVENLABS_API_KEY')
    
    if not api_key:
        raise ValueError("No API key provided. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter")
    
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching voice info for {voice_id}: {e}")
        if response.status_code == 404:
            return {"error": "Voice not found", "voice_id": voice_id}
        elif response.status_code == 401:
            return {"error": "Invalid API key", "voice_id": voice_id}
        else:
            return {"error": str(e), "voice_id": voice_id}

def print_voice_info(voice_data: Dict[str, Any]):
    """Print formatted voice information"""
    if "error" in voice_data:
        print(f"❌ Error for voice {voice_data.get('voice_id', 'unknown')}: {voice_data['error']}")
        return
    
    print(f"🎤 Voice ID: {voice_data.get('voice_id', 'N/A')}")
    print(f"📝 Name: {voice_data.get('name', 'N/A')}")
    print(f"📂 Category: {voice_data.get('category', 'N/A')}")
    
    # Labels (gender, age, accent, etc.)
    labels = voice_data.get('labels', {})
    if labels:
        print("🏷️  Labels:")
        for key, value in labels.items():
            print(f"   • {key.title()}: {value}")
    
    # Description
    description = voice_data.get('description', '')
    if description:
        print(f"📖 Description: {description}")
    
    # Preview URL
    preview_url = voice_data.get('preview_url', '')
    if preview_url:
        print(f"🔊 Preview URL: {preview_url}")
    
    # Sharing info (if available)
    sharing = voice_data.get('sharing', {})
    if sharing and sharing.get('status') == 'enabled':
        print("🌐 Shared in Voice Library: Yes")
        if sharing.get('name'):
            print(f"   • Library Name: {sharing['name']}")
        if sharing.get('description'):
            print(f"   • Library Description: {sharing['description']}")
    
    # Available for tiers
    tiers = voice_data.get('available_for_tiers', [])
    if tiers:
        print(f"💎 Available for tiers: {', '.join(tiers)}")
    
    print("-" * 60)

def main():
    """Main function to query voice information"""
    # The voice IDs we want to check
    voice_ids = [
        "XBDAUT8ybuJTTCoOLSUj",  # Male voice
        "jpmnSYDOADVEpZksbLmc"   # Female voice
    ]
    
    print("🔍 Querying ElevenLabs API for voice information...")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("⚠️  Warning: No ELEVENLABS_API_KEY environment variable found.")
        print("You can:")
        print("1. Set the environment variable: export ELEVENLABS_API_KEY='your_key_here'")
        print("2. Or provide it when running the script")
        print()
        
        # Try to get API key from user input
        try:
            api_key = input("Enter your ElevenLabs API key (or press Enter to skip): ").strip()
            if not api_key:
                print("No API key provided. Cannot proceed.")
                return
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    # Query each voice
    for i, voice_id in enumerate(voice_ids, 1):
        print(f"\n{i}. Querying voice: {voice_id}")
        voice_info = get_voice_info(voice_id, api_key)
        print_voice_info(voice_info)

if __name__ == "__main__":
    main() 