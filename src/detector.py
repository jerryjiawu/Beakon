import pyaudio
import numpy as np
from scipy import signal
import soundfile as sf
import threading
import time
import queue
import datetime
import os
import sys
import librosa
import math

# Add the birdnet directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'birdnet'))

# Import BirdNET analysis functions
try:
    import tflite_runtime.interpreter as tflite
    print("Using TensorFlow Lite Runtime")
except ImportError:
    try:
        from tensorflow import lite as tflite
        print("Using TensorFlow Lite from TensorFlow")
    except ImportError:
        print("Warning: TensorFlow Lite not available. Bird detection will be disabled.")
        tflite = None

class AcousticDetector:
    def __init__(self, 
                 sample_rate=44100,
                 chunk_size=1024,
                 channels=1,
                 low_freq=1000,
                 high_freq=8000,
                 spike_threshold=2.0,
                 recording_duration=10.0,
                 baseline_window=5.0,
                 min_bird_confidence=0.6):

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.spike_threshold = spike_threshold
        self.recording_duration = recording_duration
        self.baseline_window = baseline_window
        self.min_bird_confidence = min_bird_confidence
        
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
        
        # Bird detection setup
        self.bird_model = None
        self.bird_classes = []
        self.input_layer_index = None
        self.output_layer_index = None
        self.mdata_input_index = None
        self._load_bird_model()
        
        print(f"Acoustic Detector initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Bandpass filter: {low_freq}-{high_freq} Hz")
        print(f"  Spike threshold: {spike_threshold}x above baseline")
        print(f"  Recording duration: {recording_duration} seconds")
        print(f"  Bird confidence threshold: {min_bird_confidence}")
    
    def apply_filter(self, data):
        try:
            filtered_data = signal.filtfilt(self.b, self.a, data)
            return filtered_data
        except Exception as e:
            print(f"Filter error: {e}")
            return data
    
    def calculate_rms_amplitude(self, data):
        return np.sqrt(np.mean(data**2))
    
    def _load_bird_model(self):
        """Load the BirdNET model for species identification."""
        if tflite is None:
            print("  Bird detection: DISABLED (TensorFlow Lite not available)")
            return
            
        try:
            # Path to the BirdNET model
            model_path = os.path.join(os.path.dirname(__file__), 'birdnet', 'model', 'BirdNET_6K_GLOBAL_MODEL.tflite')
            labels_path = os.path.join(os.path.dirname(__file__), 'birdnet', 'model', 'labels.txt')
            
            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                print(f"  Bird detection: DISABLED (Model files not found)")
                return
            
            # Load TFLite model
            self.bird_model = tflite.Interpreter(model_path=model_path)
            self.bird_model.allocate_tensors()
            
            # Get input and output tensor details
            input_details = self.bird_model.get_input_details()
            output_details = self.bird_model.get_output_details()
            
            self.input_layer_index = input_details[0]['index']
            self.mdata_input_index = input_details[1]['index']
            self.output_layer_index = output_details[0]['index']
            
            # Load class labels
            with open(labels_path, 'r') as f:
                self.bird_classes = [line.strip() for line in f.readlines()]
            
            print(f"  Bird detection: ENABLED ({len(self.bird_classes)} species)")
            
        except Exception as e:
            print(f"  Bird detection: DISABLED (Error loading model: {e})")
            self.bird_model = None
    
    def _split_audio_for_analysis(self, audio_data, sample_rate=48000, chunk_duration=3.0, overlap=0.0):
        """Split audio into 3-second chunks for BirdNET analysis."""
        # Resample to 48kHz if necessary (BirdNET expects 48kHz)
        if self.sample_rate != sample_rate:
            try:
                audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=sample_rate)
            except:
                # Fallback: simple linear interpolation
                ratio = sample_rate / self.sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(np.linspace(0, len(audio_data)-1, new_length), 
                                     np.arange(len(audio_data)), audio_data)
        
        chunks = []
        chunk_samples = int(chunk_duration * sample_rate)
        step_samples = int((chunk_duration - overlap) * sample_rate)
        
        for i in range(0, len(audio_data), step_samples):
            chunk = audio_data[i:i + chunk_samples]
            
            # Skip if chunk is too short
            if len(chunk) < int(1.5 * sample_rate):  # Minimum 1.5 seconds
                break
                
            # Pad with zeros if needed
            if len(chunk) < chunk_samples:
                padded_chunk = np.zeros(chunk_samples)
                padded_chunk[:len(chunk)] = chunk
                chunk = padded_chunk
                
            chunks.append(chunk)
        
        return chunks, sample_rate
    
    def _convert_metadata(self, lat=-1, lon=-1, week=-1):
        """Convert metadata for BirdNET model."""
        m = np.array([lat, lon, week], dtype=float)
        
        # Convert week to cosine
        if m[2] >= 1 and m[2] <= 48:
            m[2] = math.cos(math.radians(m[2] * 7.5)) + 1 
        else:
            m[2] = -1

        # Add binary mask
        mask = np.ones((3,))
        if m[0] == -1 or m[1] == -1:
            mask = np.zeros((3,))
        if m[2] == -1:
            mask[2] = 0.0

        return np.concatenate([m, mask])
    
    def _custom_sigmoid(self, x, sensitivity=1.0):
        """Apply custom sigmoid function to predictions."""
        return 1 / (1.0 + np.exp(-sensitivity * x))
    
    def _predict_species(self, audio_chunk, metadata, sensitivity=1.0):
        """Predict bird species for a single audio chunk."""
        if self.bird_model is None:
            return []
            
        try:
            # Prepare input
            sig = np.expand_dims(audio_chunk, 0)
            mdata = np.expand_dims(metadata, 0)
            
            # Make prediction
            self.bird_model.set_tensor(self.input_layer_index, np.array(sig, dtype='float32'))
            self.bird_model.set_tensor(self.mdata_input_index, np.array(mdata, dtype='float32'))
            self.bird_model.invoke()
            prediction = self.bird_model.get_tensor(self.output_layer_index)[0]
            
            # Apply custom sigmoid
            p_sigmoid = self._custom_sigmoid(prediction, sensitivity)
            
            # Get label and scores
            p_labels = dict(zip(self.bird_classes, p_sigmoid))
            
            # Sort by score
            p_sorted = sorted(p_labels.items(), key=lambda x: x[1], reverse=True)
            
            # Filter out non-bird detections and low confidence
            filtered_results = []
            for species, confidence in p_sorted[:10]:  # Top 10 only
                if (species not in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise'] 
                    and confidence >= self.min_bird_confidence):
                    # Split scientific and common name
                    if '_' in species:
                        scientific, common = species.split('_', 1)
                        filtered_results.append({
                            'scientific_name': scientific,
                            'common_name': common,
                            'confidence': float(confidence)
                        })
            
            return filtered_results
            
        except Exception as e:
            print(f"Error in species prediction: {e}")
            return []
    
    def analyze_recording_for_birds(self, audio_data, lat=-1, lon=-1, week=-1, sensitivity=1.0, min_confidence=0.6):
        """
        Analyze the recorded audio for bird species.
        
        Args:
            audio_data: Numpy array of audio data
            lat: Latitude of recording location (-1 to ignore)
            lon: Longitude of recording location (-1 to ignore) 
            week: Week of year (1-48, -1 to ignore)
            sensitivity: Detection sensitivity (0.5-1.5)
            min_confidence: Minimum confidence threshold (0.01-0.99)
            
        Returns:
            dict: Dictionary with detected species and their confidence scores
        """
        if self.bird_model is None:
            return {"error": "Bird detection model not available"}
        
        try:
            # Split audio into analysis chunks
            chunks, sample_rate = self._split_audio_for_analysis(audio_data)
            
            if not chunks:
                return {"error": "Audio too short for analysis"}
            
            # Prepare metadata
            metadata = self._convert_metadata(lat, lon, week)
            
            # Analyze each chunk
            all_detections = {}
            chunk_duration = 3.0  # seconds
            
            for i, chunk in enumerate(chunks):
                start_time = i * chunk_duration
                end_time = start_time + chunk_duration
                
                detections = self._predict_species(chunk, metadata, sensitivity)
                
                # Filter by minimum confidence
                filtered_detections = [d for d in detections if d['confidence'] >= min_confidence]
                
                if filtered_detections:
                    time_key = f"{start_time:.1f}-{end_time:.1f}s"
                    all_detections[time_key] = filtered_detections
            
            # Summarize results
            species_summary = {}
            total_detections = 0
            
            for time_segment, detections in all_detections.items():
                for detection in detections:
                    species = detection['common_name']
                    confidence = detection['confidence']
                    
                    if species not in species_summary:
                        species_summary[species] = {
                            'max_confidence': confidence,
                            'avg_confidence': confidence,
                            'detection_count': 1,
                            'scientific_name': detection['scientific_name']
                        }
                    else:
                        # Update summary
                        prev_avg = species_summary[species]['avg_confidence']
                        prev_count = species_summary[species]['detection_count']
                        
                        species_summary[species]['max_confidence'] = max(
                            species_summary[species]['max_confidence'], confidence
                        )
                        species_summary[species]['avg_confidence'] = (
                            (prev_avg * prev_count + confidence) / (prev_count + 1)
                        )
                        species_summary[species]['detection_count'] += 1
                    
                    total_detections += 1
            
            return {
                'analysis_duration': len(chunks) * chunk_duration,
                'total_detections': total_detections,
                'unique_species': len(species_summary),
                'species_detected': species_summary,
                'detailed_timeline': all_detections
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _get_current_week(self):
        """Get the current week of the year (1-48, BirdNET format)."""
        current_date = datetime.datetime.now()
        week_of_year = current_date.isocalendar()[1]  # ISO week number
        # Convert to BirdNET format (4 weeks per month, max 48)
        return min(48, max(1, week_of_year))
    
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
            duration = len(full_recording) / self.sample_rate
            
            print(f"🔍 ANALYZING RECORDING ({duration:.1f}s)...")
            
            # Analyze recording for bird species
            bird_analysis = self.analyze_recording_for_birds(
                full_recording,
                lat=-1,  # Can be configured via settings
                lon=-1,  # Can be configured via settings
                week=self._get_current_week(),  # Auto-calculate current week
                sensitivity=1.0,
                min_confidence=self.min_bird_confidence
            )
            
            # Display results
            if 'error' in bird_analysis:
                print(f"❌ Bird analysis failed: {bird_analysis['error']}")
            else:
                unique_species = bird_analysis['unique_species']
                total_detections = bird_analysis['total_detections']
                
                if unique_species > 0:
                    print(f"🐦 BIRDS DETECTED: {unique_species} species, {total_detections} total detections")
                    
                    # Show top species by confidence
                    species_data = bird_analysis['species_detected']
                    sorted_species = sorted(
                        species_data.items(), 
                        key=lambda x: x[1]['max_confidence'], 
                        reverse=True
                    )
                    
                    for i, (common_name, data) in enumerate(sorted_species[:5]):  # Top 5
                        confidence = data['max_confidence']
                        count = data['detection_count']
                        scientific = data['scientific_name']
                        print(f"  {i+1}. {common_name} ({scientific}) - {confidence:.3f} confidence ({count} detections)")
                        
                    # Show timeline if there are multiple detections
                    if len(bird_analysis['detailed_timeline']) > 1:
                        print("  Timeline:")
                        for time_segment, detections in bird_analysis['detailed_timeline'].items():
                            species_names = [d['common_name'] for d in detections]
                            print(f"    {time_segment}: {', '.join(species_names)}")
                else:
                    print("🔍 No bird species detected in recording")
            
            print(f"✅ Recording analysis complete")
        
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