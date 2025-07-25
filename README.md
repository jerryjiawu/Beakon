# Acoustic Detection Setup

This script monitors live microphone input and automatically records audio when sudden increases (spikes) in noise are detected within a specific frequency range.

## Features

- **Bandpass filtering**: Filters audio to keep only frequencies between 1 kHz and 8 kHz
- **Spike detection**: Monitors for sudden increases in audio amplitude
- **Automatic recording**: Records 10 seconds of audio when a spike is detected
- **Real-time processing**: Processes audio in real-time with minimal latency

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues installing `pyaudio` on Windows, you may need to:
- Install Visual Studio Build Tools
- Or use: `pip install pipwin` then `pipwin install pyaudio`

## Usage

Run the script:
```bash
python acoustic_detector.py
```

The script will:
1. Start monitoring your default microphone
2. Apply a 1-8 kHz bandpass filter to the audio
3. Calculate a baseline noise level
4. Detect spikes (sudden increases) in audio amplitude
5. Automatically record 10 seconds of audio when a spike is detected
6. Save recordings to the `recordings/` folder

Press `Ctrl+C` to stop monitoring.

## Configuration

You can modify the detection parameters by editing the values in the `main()` function:

- `sample_rate`: Audio sampling rate (default: 44100 Hz)
- `low_freq`/`high_freq`: Bandpass filter range (default: 1000-8000 Hz)
- `spike_threshold`: Sensitivity for spike detection (default: 2.0)
- `recording_duration`: How long to record after spike detection (default: 10.0 seconds)

## Output

Recordings are saved as WAV files in the `recordings/` directory with timestamps:
- Format: `spike_recording_YYYYMMDD_HHMMSS.wav`
- Sample rate: 44.1 kHz
- Channels: Mono

## Troubleshooting

- **No microphone detected**: Check that your microphone is connected and set as the default input device
- **Permission errors**: Ensure the application has permission to access your microphone
- **PyAudio installation issues**: See installation notes above for Windows-specific solutions
