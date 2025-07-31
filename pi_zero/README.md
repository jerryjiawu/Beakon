# Acoustic Detection - Pi Zero W Edition

This is the optimized version for Raspberry Pi Zero W with limited resources.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System:**
   ```bash
   python launcher.py
   ```

3. **Access Web Interface:**
   - Open http://[pi-ip]:5000 in your browser
   - Default password: `admin123`

## Optimizations for Pi Zero W

- **Reduced Sample Rate:** 22050 Hz (vs 44100 Hz)
- **Smaller Chunks:** 512 samples (vs 1024)
- **Shorter Recordings:** 8 seconds (vs 10 seconds)
- **Higher Confidence:** 0.7 threshold (vs 0.6)
- **Memory Efficient:** Max 5 detections stored (vs 10)
- **16-bit Audio:** Saves storage space
- **Bird-only Storage:** Only saves recordings with confirmed birds

## Hardware Requirements

- Raspberry Pi Zero W (512MB RAM, single-core ARM)
- USB microphone or USB sound card
- MicroSD card (8GB minimum)

## Features

- 🎤 **Real-time Detection:** Continuous audio monitoring
- 🐦 **Bird Identification:** AI-powered species recognition
- 🌐 **Web Dashboard:** Simple, responsive interface
- 🔊 **Audio Playback:** Stream and download recordings
- ⚙️ **Settings:** Adjustable detection parameters
- 📱 **Mobile Friendly:** Works on phones and tablets

## File Structure

```
pi_zero/
├── launcher.py          # Main application
├── requirements.txt     # Dependencies
├── templates/          # Web interface
│   ├── pi_login.html
│   ├── pi_dashboard.html
│   └── pi_settings.html
└── recordings/         # Audio files (created automatically)
```

## Network Access

The system automatically detects your Pi's IP address and displays it on startup:

```
🌐 Local:    http://localhost:5000
🌍 Network:  http://192.168.1.xxx:5000
🔑 Password: admin123
```

## Troubleshooting

- **Audio Issues:** Check microphone permissions and USB connections
- **Memory Errors:** Increase swap file size or reduce detection parameters
- **Network Access:** Ensure Pi is connected to WiFi
- **Performance:** Monitor CPU usage and adjust settings accordingly
