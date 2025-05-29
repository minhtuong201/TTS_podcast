#!/usr/bin/env python3
"""
Setup script for TTS Podcast Pipeline
Helps configure environment and install dependencies
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✓ Python {sys.version.split()[0]} is compatible")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✓ Virtual environment created")
        
        print("\n💡 To activate the virtual environment, run:")
        if os.name == 'nt':  # Windows
            print("   .venv\\Scripts\\activate")
        else:  # Linux/macOS
            print("   source .venv/bin/activate")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📥 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment_file():
    """Create .env file from template"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            shutil.copy2(env_example, env_file)
            print("✓ Created .env file from template")
            print("\n💡 Edit .env file to add your API keys:")
            print("   - OPENROUTER_API_KEY (required for summarization)")
            print("   - ELEVENLABS_API_KEY (for ElevenLabs TTS)")
            print("   - OPENAI_API_KEY (for OpenAI TTS)")
            print("   - AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (for Azure TTS)")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("❌ env_example.txt not found")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["output", "src/tts/voices"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path("src")))
        
        import pdf_ingest
        import lang_detect
        import summarizer
        import script_gen
        import audio_mixer
        
        print("✓ Core modules imported successfully")
        
        # Test TTS backends
        from main import get_available_tts_backends
        backends = get_available_tts_backends()
        
        if backends:
            print(f"✓ Available TTS backends: {', '.join(backends)}")
        else:
            print("⚠️  No TTS backends available (API keys needed)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print_header("🚀 Setup Complete!")
    
    print("""
Next steps:

1. 📝 Configure API Keys:
   Edit the .env file and add your API keys:
   
   # Required for summarization and script generation
   OPENROUTER_API_KEY=your_key_here
   
   # At least one TTS provider (choose one or more):
   ELEVENLABS_API_KEY=your_key_here      # Recommended for quality
   OPENAI_API_KEY=your_key_here          # Alternative TTS
   AZURE_SPEECH_KEY=your_key_here        # Enterprise option
   # Coqui XTTS is free but requires: pip install TTS torch

2. 🧪 Test the Pipeline:
   python test_sample.py

3. 🎙️ Generate Your First Podcast:
   python src/main.py your_document.pdf

4. 📖 Read the Documentation:
   Check README.md for detailed usage instructions

""")

    print("📚 API Key Resources:")
    print("   • OpenRouter: https://openrouter.ai")
    print("   • ElevenLabs: https://elevenlabs.io")
    print("   • OpenAI: https://platform.openai.com")
    print("   • Azure Speech: https://azure.microsoft.com/services/cognitive-services/speech-services/")

def main():
    """Main setup function"""
    print_header("TTS Podcast Pipeline Setup")
    
    success_count = 0
    total_steps = 6
    
    # Step 1: Check Python version
    if check_python_version():
        success_count += 1
    
    # Step 2: Setup virtual environment
    if setup_virtual_environment():
        success_count += 1
    
    # Step 3: Install dependencies
    if install_dependencies():
        success_count += 1
    
    # Step 4: Setup environment file
    if setup_environment_file():
        success_count += 1
    
    # Step 5: Create directories
    create_directories()
    success_count += 1
    
    # Step 6: Test installation
    if test_installation():
        success_count += 1
    
    # Summary
    print(f"\n📊 Setup Summary: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        display_next_steps()
    else:
        print("\n⚠️  Setup completed with some issues. Check the output above for details.")
        print("💡 You may need to manually configure some components.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc() 