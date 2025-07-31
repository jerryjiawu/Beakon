#!/bin/bash
# fix_numpy.sh - Quick fix for NumPy OpenBLAS issue on Pi Zero W

echo "🔧 Fixing NumPy OpenBLAS issue..."

# Remove broken numpy installation
echo "🗑️  Removing broken NumPy installation..."
pip3 uninstall -y numpy scipy 2>/dev/null || true

# Install system dependencies
echo "📦 Installing OpenBLAS dependencies..."
sudo apt-get update
sudo apt-get install -y libopenblas-dev libatlas-base-dev gfortran

# Install system NumPy/SciPy (more reliable)
echo "📊 Installing system NumPy and SciPy..."
sudo apt-get install -y python3-numpy python3-scipy

# Test the installation
echo "🧪 Testing NumPy installation..."
python3 -c "import numpy; print(f'✅ NumPy {numpy.__version__} working!')" || {
    echo "⚠️  System packages failed, trying pip with older version..."
    pip3 install --no-cache-dir numpy==1.21.6 scipy==1.7.3
}

echo "✅ NumPy fix complete!"
echo "🎯 Try running your application again"
