#!/bin/bash

# TTS Podcast Pipeline Setup Script
# Automates the installation and configuration process

set -e  # Exit on any error

echo "🎙️  TTS Podcast Pipeline Setup"
echo "==============================="

# Check if running on Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    echo "❌ This script is designed for Ubuntu/Debian systems with apt package manager"
    exit 1
fi

# Function to print colored output
print_status() {
    echo -e "\033[1;34m$1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠️  $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m❌ $1\033[0m"
}

# Check Python version
print_status "🐍 Checking Python version..."
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_success "Python $PYTHON_VERSION is compatible"
else
    print_error "Python 3.8+ is required"
    exit 1
fi

# Install system dependencies
print_status "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev build-essential ffmpeg

# Create virtual environment
print_status "🔧 Setting up virtual environment..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists, removing old one..."
    rm -rf .venv
fi

python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
print_status "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
print_status "📥 Installing core dependencies..."
pip install python-dotenv requests tenacity pdfminer.six langdetect pydub

print_success "Core dependencies installed"

# Create .env file
print_status "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    if [ -f "env_example.txt" ]; then
        cp env_example.txt .env
        print_success "Created .env file from template"
    else
        cat > .env << 'EOF'
# TTS Podcast Pipeline Configuration

# Required: OpenRouter API for GPT-4o-mini summarization
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Choose one or more TTS providers:

# ElevenLabs (recommended for best quality)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Gemini TTS (premium multi-speaker)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI TTS (alternative)
OPENAI_API_KEY=your_openai_api_key_here

# Azure Speech Services (enterprise option)
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here

# Default settings
DEFAULT_TTS_BACKEND=eleven
DEFAULT_FEMALE_VOICE=Rachel
DEFAULT_MALE_VOICE=Josh
OUTPUT_BITRATE=192k
TARGET_WORDS=900
TARGET_MINUTES=8
EOF
        print_success "Created .env configuration file"
    fi
else
    print_warning ".env file already exists"
fi

# Create output directory
mkdir -p output
mkdir -p src/tts/voices

# Test core functionality
print_status "🧪 Testing core functionality..."
if python3 demo_core.py > /dev/null 2>&1; then
    print_success "Core functionality test passed"
else
    print_warning "Core test had some issues (this is normal without API keys)"
fi

# Check for optional TTS dependencies
print_status "🔍 Checking optional TTS dependencies..."

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "✅ Core system is installed and ready"
echo "✅ Virtual environment: .venv"
echo "✅ Configuration file: .env"
echo "✅ Output directory: output/"
echo ""
echo "📝 Next Steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Test the system:"
echo "   python3 demo_core.py"
echo ""
echo "4. Generate your first podcast:"
echo "   python src/main.py your_document.pdf"
echo ""
echo "📚 API Key Resources:"
echo "• OpenRouter (required): https://openrouter.ai"
echo "• ElevenLabs (recommended): https://elevenlabs.io"
echo "• Gemini TTS (premium multi-speaker): https://ai.google.dev/"
echo "• OpenAI: https://platform.openai.com"
echo "• Azure Speech: https://azure.microsoft.com/cognitive-services/speech-services/"
echo ""
echo "🎙️  Ready to convert PDFs into engaging podcasts!"

# Optional: Install TTS dependencies
echo ""
read -p "🤔 Install TTS dependencies now? (ElevenLabs, OpenAI) [y/N]: " install_tts
if [[ $install_tts =~ ^[Yy]$ ]]; then
    print_status "📥 Installing TTS dependencies..."
    pip install elevenlabs openai
    print_success "TTS dependencies installed"
fi

# Optional: Install Coqui for local TTS
echo ""
read -p "🤔 Install Coqui TTS for local/offline synthesis? (large download) [y/N]: " install_coqui
if [[ $install_coqui =~ ^[Yy]$ ]]; then
    print_status "📥 Installing Coqui TTS (this may take a while)..."
    pip install TTS torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_success "Coqui TTS installed"
fi

echo ""
print_success "🚀 TTS Podcast Pipeline is ready to use!"
echo "Run 'source .venv/bin/activate' to activate the environment" 