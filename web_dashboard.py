from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import json
import os
import datetime
import threading
import time
import numpy as np
import plotly.graph_objs as go
import plotly.utils
from src.detector import AcousticDetector
from src.settings import Settings
import soundfile as sf

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this!

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Load password from settings
def get_system_password():
    """Get the system password from settings."""
    settings = Settings()
    settings.load()
    return settings.config.get('web', {}).get('password', 'admin123')

@login_manager.user_loader
def load_user(user_id):
    if user_id == "authenticated_user":
        return User(user_id)
    return None

# Global variables for the detector and detection history
detector = None
detection_history = []
detector_thread = None
detector_running = False

class DetectionStore:
    """Store detection results in memory."""
    def __init__(self):
        self.detections = []
        self.max_detections = 100  # Keep last 100 detections
    
    def add_detection(self, detection_data):
        self.detections.insert(0, detection_data)  # Add to beginning
        if len(self.detections) > self.max_detections:
            self.detections = self.detections[:self.max_detections]
    
    def get_recent_detections(self, limit=20):
        return self.detections[:limit]
    
    def clear_detections(self):
        self.detections = []

detection_store = DetectionStore()

# Custom detector class that stores detections
class WebAcousticDetector(AcousticDetector):
    def process_recording(self):
        if self.recording_data:
            timestamp = datetime.datetime.now()
            
            full_recording = np.concatenate(self.recording_data)
            duration = len(full_recording) / self.sample_rate
            
            print(f"🔍 ANALYZING RECORDING ({duration:.1f}s)...")
            
            # Analyze recording for bird species
            bird_analysis = self.analyze_recording_for_birds(
                full_recording,
                lat=-1,  # Can be configured via settings
                lon=-1,  # Can be configured via settings
                week=self._get_current_week(),
                sensitivity=1.0,
                min_confidence=self.min_bird_confidence
            )
            
            # Store detection in web store
            detection_data = {
                'timestamp': timestamp.isoformat(),
                'duration': duration,
                'analysis': bird_analysis,
                'formatted_time': timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            detection_store.add_detection(detection_data)
            
            # Display results (console)
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
                else:
                    print("🔍 No bird species detected in recording")
            
            print(f"✅ Recording analysis complete")
        
        # Reset recording state
        self.is_recording = False
        self.recording_start_time = None
        self.recording_data = []

def run_detector():
    """Run the acoustic detector in a separate thread."""
    global detector, detector_running
    try:
        settings = Settings()
        settings.load()
        
        # Create detector with settings
        detector_config = settings.config.get('detector', {})
        detector = WebAcousticDetector(
            sample_rate=detector_config.get('sample_rate', 44100),
            chunk_size=detector_config.get('chunk_size', 1024),
            channels=detector_config.get('channels', 1),
            low_freq=detector_config.get('low_freq', 1000),
            high_freq=detector_config.get('high_freq', 8000),
            spike_threshold=detector_config.get('spike_threshold', 2.0),
            recording_duration=detector_config.get('recording_duration', 10.0),
            baseline_window=detector_config.get('baseline_window', 5.0),
            min_bird_confidence=detector_config.get('min_bird_confidence', 0.6)
        )
        
        detector_running = True
        detector.start_monitoring()
    except Exception as e:
        print(f"Detector error: {e}")
        detector_running = False

@app.route('/')
@login_required
def dashboard():
    """Main dashboard page."""
    recent_detections = detection_store.get_recent_detections(10)
    
    # Calculate statistics
    total_detections = len(detection_store.detections)
    species_count = set()
    for detection in detection_store.detections:
        if 'analysis' in detection and 'species_detected' in detection['analysis']:
            species_count.update(detection['analysis']['species_detected'].keys())
    
    stats = {
        'total_detections': total_detections,
        'unique_species': len(species_count),
        'detector_status': 'Running' if detector_running else 'Stopped',
        'last_detection': recent_detections[0]['formatted_time'] if recent_detections else 'None'
    }
    
    return render_template('dashboard.html', detections=recent_detections, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        password = request.form['password']
        system_password = get_system_password()
        
        if password == system_password:
            user = User("authenticated_user")
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user."""
    logout_user()
    return redirect(url_for('login'))

@app.route('/settings')
@login_required
def settings():
    """Settings editor page."""
    settings = Settings()
    settings.load()
    return render_template('settings.html', settings=settings.config)

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def api_settings():
    """API endpoint for settings management."""
    settings = Settings()
    
    if request.method == 'GET':
        settings.load()
        return jsonify(settings.config)
    
    elif request.method == 'POST':
        try:
            new_settings = request.json
            settings.config = new_settings
            settings.save()
            return jsonify({'status': 'success', 'message': 'Settings saved successfully'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/analyzer')
@login_required
def analyzer():
    """Audio analyzer page."""
    return render_template('analyzer.html')

@app.route('/api/analyze_audio', methods=['POST'])
@login_required
def analyze_audio():
    """Analyze uploaded audio file."""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # Load and analyze audio
        audio_data, sample_rate = sf.read(temp_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Convert to mono
        
        # Create temporary detector for analysis
        temp_detector = AcousticDetector()
        if temp_detector.bird_model is not None:
            analysis = temp_detector.analyze_recording_for_birds(audio_data)
        else:
            analysis = {'error': 'Bird detection model not available'}
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections')
@login_required
def api_detections():
    """Get recent detections as JSON."""
    limit = request.args.get('limit', 20, type=int)
    detections = detection_store.get_recent_detections(limit)
    return jsonify(detections)

@app.route('/api/detector/start', methods=['POST'])
@login_required
def start_detector():
    """Start the acoustic detector."""
    global detector_thread, detector_running
    
    if not detector_running:
        detector_thread = threading.Thread(target=run_detector, daemon=True)
        detector_thread.start()
        return jsonify({'status': 'success', 'message': 'Detector started'})
    else:
        return jsonify({'status': 'info', 'message': 'Detector already running'})

@app.route('/api/detector/stop', methods=['POST'])
@login_required
def stop_detector():
    """Stop the acoustic detector."""
    global detector, detector_running
    
    if detector and detector_running:
        detector_running = False
        detector.stop_monitoring()
        return jsonify({'status': 'success', 'message': 'Detector stopped'})
    else:
        return jsonify({'status': 'info', 'message': 'Detector not running'})

@app.route('/api/detections/clear', methods=['POST'])
@login_required
def clear_detections():
    """Clear all detection history."""
    detection_store.clear_detections()
    return jsonify({'status': 'success', 'message': 'Detection history cleared'})

@app.route('/api/statistics')
@login_required
def api_statistics():
    """Get detection statistics."""
    detections = detection_store.detections
    
    # Species frequency chart data
    species_freq = {}
    daily_counts = {}
    
    for detection in detections:
        # Count species
        if 'analysis' in detection and 'species_detected' in detection['analysis']:
            for species in detection['analysis']['species_detected'].keys():
                species_freq[species] = species_freq.get(species, 0) + 1
        
        # Count daily detections
        date = detection['timestamp'][:10]  # Get date part
        daily_counts[date] = daily_counts.get(date, 0) + 1
    
    # Create Plotly charts
    species_chart = {
        'data': [{
            'x': list(species_freq.keys()),
            'y': list(species_freq.values()),
            'type': 'bar',
            'name': 'Species Detections'
        }],
        'layout': {
            'title': 'Most Detected Species',
            'xaxis': {'title': 'Species'},
            'yaxis': {'title': 'Detection Count'}
        }
    }
    
    daily_chart = {
        'data': [{
            'x': list(daily_counts.keys()),
            'y': list(daily_counts.values()),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Daily Detections'
        }],
        'layout': {
            'title': 'Detections Over Time',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Detection Count'}
        }
    }
    
    return jsonify({
        'species_chart': species_chart,
        'daily_chart': daily_chart,
        'total_detections': len(detections),
        'unique_species': len(species_freq)
    })

if __name__ == '__main__':
    # Create default settings if they don't exist
    settings = Settings()
    settings.load()
    if not settings.config:
        settings.config = {
            'detector': {
                'sample_rate': 44100,
                'chunk_size': 1024,
                'channels': 1,
                'low_freq': 1000,
                'high_freq': 8000,
                'spike_threshold': 2.0,
                'recording_duration': 10.0,
                'baseline_window': 5.0,
                'min_bird_confidence': 0.6
            },
            'web': {
                'host': '127.0.0.1',
                'port': 5000,
                'debug': True,
                'password': 'admin123'
            }
        }
        settings.save()
    
    # Run Flask app
    web_config = settings.config.get('web', {})
    app.run(
        host=web_config.get('host', '127.0.0.1'),
        port=web_config.get('port', 5000),
        debug=web_config.get('debug', True)
    )
