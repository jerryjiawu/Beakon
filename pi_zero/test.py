#!/usr/bin/env python3
"""
test_install.py - Verify Pi Zero W installation
Run this after install.sh to check if everything works
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module."""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {e}")
        return False

def test_audio():
    """Test audio functionality."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        print(f"✅ Audio system: {device_count} devices found")
        return True
    except Exception as e:
        print(f"❌ Audio system: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow/TensorFlow Lite."""
    try:
        import tflite_runtime.interpreter as tflite
        print("✅ TensorFlow Lite runtime")
        return True
    except ImportError:
        try:
            import tensorflow as tf
            print(f"✅ TensorFlow {tf.__version__}")
            return True
        except ImportError:
            print("❌ No TensorFlow or TensorFlow Lite found")
            return False
    except Exception as e:
        print(f"⚠️  TensorFlow error: {e}")
        return False

def test_birdnet_model():
    """Test if BirdNET model is accessible."""
    import os
    model_path = "../src/birdnet/model/BirdNET_6K_GLOBAL_MODEL.tflite"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"✅ BirdNET model found ({size_mb:.1f}MB)")
        return True
    else:
        print(f"❌ BirdNET model not found at {model_path}")
        return False

def check_memory():
    """Check available memory."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        total_mb = memory.total / 1024 / 1024
        print(f"💾 Memory: {available_mb:.0f}MB available / {total_mb:.0f}MB total")
        
        if available_mb < 100:
            print("⚠️  Low memory warning! Consider rebooting or closing other programs")
        return True
    except Exception as e:
        print(f"❌ Memory check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Pi Zero W Installation")
    print("=" * 40)
    
    tests = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"), 
        ("Flask", "Flask web framework"),
        ("flask_login", "Flask-Login"),
        ("soundfile", "SoundFile"),
        ("librosa", "Librosa audio processing"),
        ("psutil", "System utilities"),
    ]
    
    passed = 0
    total = len(tests) + 4  # +4 for special tests
    
    # Test basic imports
    for module, description in tests:
        if test_import(module, description):
            passed += 1
    
    # Test special functionality
    if test_audio():
        passed += 1
    if test_tensorflow():
        passed += 1
    if test_birdnet_model():
        passed += 1
    if check_memory():
        passed += 1
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Pi Zero W is ready!")
        print("🚀 Run: python3 launch.py")
    elif passed >= total - 2:
        print("⚠️  Most tests passed. System should work with minor issues.")
    else:
        print("❌ Multiple issues found. Check the errors above.")
        print("💡 Try running install.sh again or check the README")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
