import pyaudio
import numpy as np
from scipy import signal
import soundfile as sf
import threading
import time
import queue
import datetime
import os

class AcousticDetector:
    def __init__(self, 
                 sample_rate=44100,
                 chunk_size=1024,
                 channels=1,
                 low_freq=1000,
                 high_freq=8000,
                 spike_threshold=2.0,
                 recording_duration=10.0,
                 baseline_window=5.0):

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.spike_threshold = spike_threshold
        self.recording_duration = recording_duration
        self.baseline_window = baseline_window
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        
        self.baseline_samples = int(baseline_window * sample_rate / chunk_size)
        self.amplitude_history = queue.deque(maxlen=self.baseline_samples)

        self.is_recording = False
        self.recording_start_time = None
        self.recording_data = []
        
        print(f"Acoustic Detector initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Bandpass filter: {low_freq}-{high_freq} Hz")
        print(f"  Spike threshold: {spike_threshold}x above baseline")
        print(f"  Recording duration: {recording_duration} seconds")
    
    def apply_filter(self, data):
        try:
            filtered_data = signal.filtfilt(self.b, self.a, data)
            return filtered_data
        except Exception as e:
            print(f"Filter error: {e}")
            return data
    
    def calculate_rms_amplitude(self, data):
        return np.sqrt(np.mean(data**2))
    
    def detect_spike(self, current_amplitude):
        """
        Detect if current amplitude represents a spike above baseline.
        
        Args:
            current_amplitude: Current RMS amplitude
            
        Returns:
            bool: True if spike detected, False otherwise
        """
        if len(self.amplitude_history) < self.baseline_samples:
            return False
        
        baseline_amplitudes = list(self.amplitude_history)
        mean_baseline = np.mean(baseline_amplitudes)
        std_baseline = np.std(baseline_amplitudes)
        
        threshold = mean_baseline + self.spike_threshold * std_baseline
        
        return current_amplitude > threshold
    
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_data = []
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"🔴 SPIKE DETECTED! Starting recording at {timestamp}")
    
    def process_recording(self):
        if self.recording_data:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            full_recording = np.concatenate(self.recording_data)
            
            
        
        # Reset recording state
        self.is_recording = False
        self.recording_start_time = None
        self.recording_data = []
    
    def process_audio_chunk(self, data):
        """Process a chunk of audio data."""
        audio_data = np.frombuffer(data, dtype=np.float32)
        
        # Apply bandpass filter
        filtered_data = self.apply_filter(audio_data)
        
        # Calculate RMS amplitude
        amplitude = self.calculate_rms_amplitude(filtered_data)
        
        # Check for spike detection
        if not self.is_recording and self.detect_spike(amplitude):
            self.start_recording()
        
        # Add to baseline history if not recording
        if not self.is_recording:
            self.amplitude_history.append(amplitude)
        
        # Handle recording
        if self.is_recording:
            self.recording_data.append(filtered_data)
            
            # Check if recording duration exceeded
            if time.time() - self.recording_start_time >= self.recording_duration:
                self.process_recording()
        
        # Print status every few seconds
        if len(self.amplitude_history) % 50 == 0:  # Adjust frequency as needed
            baseline_mean = np.mean(list(self.amplitude_history)) if self.amplitude_history else 0
            status = "🔴 RECORDING" if self.is_recording else "🟢 MONITORING"
            print(f"{status} | Amplitude: {amplitude:.4f} | Baseline: {baseline_mean:.4f}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio status: {status}")
        
        self.process_audio_chunk(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_monitoring(self):
        """Start the audio monitoring loop."""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            print("\n🎤 Starting acoustic monitoring...")
            print("Press Ctrl+C to stop")
            
            # Start the stream
            self.stream.start_stream()
            
            # Keep the program running
            while self.stream.is_active():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping acoustic monitoring...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the audio monitoring and clean up."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Save any ongoing recording
        if self.is_recording:
            self.process_recording()
        
        self.audio.terminate()
        print("✅ Cleanup complete")