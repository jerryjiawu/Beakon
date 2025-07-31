#!/bin/bash
# install.sh - Optimized installation for Raspberry Pi Zero W
# Run this directly on your Raspberry Pi Zero W

echo "🚀 Pi Zero W Acoustic Detection Installation"
echo "⏱️  This will take 15-30 minutes on Pi Zero W..."

# Check if running on Pi Zero W
if ! grep -q "Pi Zero" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This script is optimized for Raspberry Pi Zero W"
fi

# Configure pip to use piwheels (pre-built ARM wheels)
echo "📦 Configuring piwheels for faster installs..."
mkdir -p ~/.pip ~/.config/pip
cat > ~/.pip/pip.conf << EOF
[global]
extra-index-url=https://www.piwheels.org/simple
timeout=300
retries=3
EOF

# Also create for newer pip versions
cat > ~/.config/pip/pip.conf << EOF
[global]
extra-index-url=https://www.piwheels.org/simple
timeout=300
retries=3
EOF

# Increase swap to handle memory-intensive installs
echo "💾 Increasing swap space for installation..."
sudo dphys-swapfile swapoff 2>/dev/null || true
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile 2>/dev/null || echo "CONF_SWAPSIZE=1024" | sudo tee -a /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Upgrade pip first
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install system dependencies first (these are fast)
echo "📚 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    libasound2-dev \
    portaudio19-dev \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev \
    build-essential \
    git

# Install core packages from piwheels (much faster!)
echo "⚡ Installing core packages from piwheels..."
# Install core packages from piwheels (much faster!)
echo "⚡ Installing core packages from piwheels..."
pip3 install --no-cache-dir \
    numpy \
    Flask \
    Flask-Login \
    psutil \
    soundfile

# Install PyAudio (try multiple methods)
echo "🎤 Installing PyAudio..."
pip3 install --no-cache-dir pyaudio || {
    echo "⚠️  PyAudio pip failed, trying system package..."
    sudo apt-get install -y python3-pyaudio || {
        echo "⚠️  System package failed, building from source..."
        pip3 install --no-cache-dir --no-binary pyaudio pyaudio
    }
}

# Install scipy (can be slow, but piwheels should help)
echo "🔢 Installing scipy..."
pip3 install --no-cache-dir scipy || {
    echo "⚠️  scipy failed, trying system package..."
    sudo apt-get install -y python3-scipy
}

# Install librosa (this is the slowest one)
echo "🎵 Installing librosa (this may take 10-15 minutes)..."
pip3 install --no-cache-dir librosa==0.9.2 || {
    echo "⚠️  librosa failed, trying older version..."
    pip3 install --no-cache-dir librosa==0.8.1
}

# Install TensorFlow Lite for ARM (try multiple sources)
echo "🧠 Installing TensorFlow Lite..."
pip3 install --no-cache-dir tflite-runtime || {
    echo "⚠️  tflite-runtime failed, trying from GitHub..."
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    wget -q "https://github.com/PINTO0309/TensorflowLite-bin/releases/download/v2.12.0/tflite_runtime-2.12.0-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_armv6l.whl" -O tflite.whl 2>/dev/null && {
        pip3 install --no-cache-dir tflite.whl
        rm tflite.whl
    } || {
        echo "⚠️  TensorFlow Lite failed, installing regular TensorFlow (will be slow)..."
        pip3 install --no-cache-dir tensorflow==2.12.0
    }
}

# Restore original swap size
echo "💾 Restoring swap size..."
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Set up permissions for audio
echo "🔊 Setting up audio permissions..."
sudo usermod -a -G audio $USER

echo ""
echo "✅ Installation complete!"
echo "📊 Memory usage:"
free -h
echo ""
echo "🎯 To run the system:"
echo "   cd $(pwd)"
echo "   python3 launch.py"
echo ""
echo "🌐 Access at: http://$(hostname -I | awk '{print $1}'):5000"
echo "🔑 Default password: admin123"
echo ""
echo "⚠️  You may need to reboot for audio permissions: sudo reboot"
