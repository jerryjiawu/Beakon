#!/usr/bin/env python3
"""
Acoustic Detection System - All-in-One Launcher
Starts both the detection engine and web dashboard in a single application.
"""

import sys
import os
import socket
import threading
import time
import signal
import ftplib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import json
import datetime
import numpy as np
import plotly.graph_objs as go
import plotly.utils
from src.detector import AcousticDetector
from src.settings import Settings
import soundfile as sf

# Global variables
detector = None
detection_history = []
detector_thread = None
detector_running = False

def get_local_ip():
    """Get the local IPv4 address."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def print_startup_info(host, port):
    """Print colorful startup information."""
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🎤 ACOUSTIC DETECTION SYSTEM STARTED")
    print("="*60)
    print(f"🌐 Local Access:     http://localhost:{port}")
    print(f"🌍 Network Access:   http://{local_ip}:{port}")
    print("="*60)
    print("🔑 Default Login:")
    print("   Password: admin123")
    print("="*60)
    print("📊 Features Available:")
    print("   • Real-time bird detection")
    print("   • Live web dashboard")
    print("   • Audio file analyzer")
    print("   • Settings management")
    print("   • Detection history")
    print("="*60)
    print("⏹️  Press Ctrl+C to stop the system")
    print("="*60)

# Flask App Setup
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

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

class DetectionStore:
    """Store detection results in memory."""
    def __init__(self):
        self.detections = []
        self.max_detections = 10  # Store only last 10 bird detections
    
    def add_detection(self, detection_data):
        # Only store detections that actually have birds
        if detection_data.get('analysis', {}).get('unique_species', 0) > 0:
            self.detections.insert(0, detection_data)
            if len(self.detections) > self.max_detections:
                # Remove old audio files when removing old detections
                for old_detection in self.detections[self.max_detections:]:
                    if 'audio_file' in old_detection:
                        try:
                            os.remove(old_detection['audio_file'])
                        except:
                            pass
                self.detections = self.detections[:self.max_detections]
    
    def get_recent_detections(self, limit=10):
        return self.detections[:min(limit, len(self.detections))]
    
    def clear_detections(self):
        # Remove all audio files before clearing
        for detection in self.detections:
            if 'audio_file' in detection:
                try:
                    os.remove(detection['audio_file'])
                except:
                    pass
        self.detections = []

detection_store = DetectionStore()

class FTPUploader:
    """Handles FTP uploads and local file cleanup."""
    
    def __init__(self):
        self.upload_queue = []
        self.upload_thread = None
        self.upload_running = False
    
    def start_uploader(self):
        """Start the FTP upload thread."""
        if not self.upload_running:
            self.upload_running = True
            self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
            self.upload_thread.start()
            print("📤 FTP uploader started")
    
    def stop_uploader(self):
        """Stop the FTP upload thread."""
        self.upload_running = False
        if self.upload_thread:
            self.upload_thread.join(timeout=5)
    
    def queue_upload(self, file_path, remote_name=None):
        """Queue a file for FTP upload."""
        if not remote_name:
            remote_name = os.path.basename(file_path)
        
        self.upload_queue.append({
            'local_path': file_path,
            'remote_name': remote_name,
            'timestamp': time.time()
        })
        print(f"📤 Queued for upload: {remote_name}")
    
    def _upload_worker(self):
        """Background worker that handles FTP uploads."""
        while self.upload_running:
            try:
                settings = Settings()
                settings.load()
                ftp_config = settings.config.get('ftp', {})
                
                if not ftp_config.get('enabled', False):
                    time.sleep(10)
                    continue
                
                if not self.upload_queue:
                    time.sleep(5)
                    continue
                
                # Process upload queue
                upload_item = self.upload_queue.pop(0)
                success = self._upload_file(upload_item, ftp_config)
                
                if success:
                    print(f"✅ Uploaded: {upload_item['remote_name']}")
                    # Clean up local file after successful upload if configured
                    if ftp_config.get('upload_all', True):
                        self._cleanup_local_files(ftp_config)
                else:
                    # Re-queue failed uploads (with limit)
                    if upload_item.get('retry_count', 0) < ftp_config.get('retry_attempts', 3):
                        upload_item['retry_count'] = upload_item.get('retry_count', 0) + 1
                        self.upload_queue.append(upload_item)
                        print(f"🔄 Retrying upload: {upload_item['remote_name']} (attempt {upload_item['retry_count']})")
                    else:
                        print(f"❌ Failed to upload after retries: {upload_item['remote_name']}")
                
            except Exception as e:
                print(f"❌ Upload worker error: {e}")
                time.sleep(10)
    
    def _upload_file(self, upload_item, ftp_config):
        """Upload a single file via FTP."""
        try:
            if not os.path.exists(upload_item['local_path']):
                return False
            
            # Connect to FTP server
            ftp = ftplib.FTP()
            ftp.connect(ftp_config['host'], ftp_config.get('port', 21), timeout=ftp_config.get('timeout', 30))
            ftp.login(ftp_config['username'], ftp_config['password'])
            
            # Change to target directory (create if needed)
            target_dir = ftp_config.get('directory', '/acoustic_detection')
            try:
                ftp.cwd(target_dir)
            except ftplib.error_perm:
                # Create directory if it doesn't exist
                ftp.mkd(target_dir)
                ftp.cwd(target_dir)
            
            # Upload file
            with open(upload_item['local_path'], 'rb') as f:
                ftp.storbinary(f'STOR {upload_item["remote_name"]}', f)
            
            ftp.quit()
            return True
            
        except Exception as e:
            print(f"❌ FTP upload error: {e}")
            return False
    
    def _cleanup_local_files(self, ftp_config):
        """Clean up old local files based on retention policy."""
        try:
            keep_days = ftp_config.get('keep_local_days', 1)
            cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
            
            recordings_dir = 'recordings'
            if not os.path.exists(recordings_dir):
                return
            
            for filename in os.listdir(recordings_dir):
                file_path = os.path.join(recordings_dir, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    # Don't delete files that are still in the detection store
                    if not any(d.get('filename') == filename for d in detection_store.detections):
                        os.remove(file_path)
                        print(f"🗑️  Cleaned up old file: {filename}")
                        
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

ftp_uploader = FTPUploader()

class WebAcousticDetector(AcousticDetector):
    """Enhanced detector that stores web results."""
    def process_recording(self):
        if self.recording_data:
            timestamp = datetime.datetime.now()
            
            full_recording = np.concatenate(self.recording_data)
            duration = len(full_recording) / self.sample_rate
            
            print(f"🔍 ANALYZING RECORDING ({duration:.1f}s)...")
            
            bird_analysis = self.analyze_recording_for_birds(
                full_recording,
                lat=-1,
                lon=-1,
                week=self._get_current_week(),
                sensitivity=1.0,
                min_confidence=self.min_bird_confidence
            )
            
            # Only store and save audio if birds were detected
            if not bird_analysis.get('error') and bird_analysis.get('unique_species', 0) > 0:
                # Create recordings directory if it doesn't exist
                os.makedirs('recordings', exist_ok=True)
                
                # Save the audio file
                filename = f"detection_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
                audio_file_path = os.path.join('recordings', filename)
                
                try:
                    import soundfile as sf
                    sf.write(audio_file_path, full_recording, self.sample_rate)
                    
                    # Store detection data with audio file path
                    detection_data = {
                        'timestamp': timestamp.isoformat(),
                        'duration': duration,
                        'analysis': bird_analysis,
                        'formatted_time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        'audio_file': audio_file_path,
                        'filename': filename
                    }
                    detection_store.add_detection(detection_data)
                    
                    # Queue for FTP upload
                    ftp_uploader.queue_upload(audio_file_path, filename)
                    
                    print(f"🐦 BIRDS DETECTED: {bird_analysis['unique_species']} species")
                    print(f"💾 Audio saved: {filename}")
                    
                except Exception as e:
                    print(f"❌ Error saving audio: {e}")
                    # Store without audio file if saving fails
                    detection_data = {
                        'timestamp': timestamp.isoformat(),
                        'duration': duration,
                        'analysis': bird_analysis,
                        'formatted_time': timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    detection_store.add_detection(detection_data)
            
            # Console output
            if 'error' in bird_analysis:
                print(f"❌ Bird analysis failed: {bird_analysis['error']}")
            else:
                unique_species = bird_analysis['unique_species']
                total_detections = bird_analysis['total_detections']
                
                if unique_species > 0:
                    species_data = bird_analysis['species_detected']
                    sorted_species = sorted(
                        species_data.items(), 
                        key=lambda x: x[1]['max_confidence'], 
                        reverse=True
                    )
                    
                    for i, (common_name, data) in enumerate(sorted_species[:3]):
                        confidence = data['max_confidence']
                        scientific = data['scientific_name']
                        print(f"  {i+1}. {common_name} ({scientific}) - {confidence:.3f}")
                else:
                    print("🔍 No bird species detected (not saved)")
            
            print(f"✅ Analysis complete")
        
        self.is_recording = False
        self.recording_start_time = None
        self.recording_data = []

def run_detector():
    """Run the acoustic detector."""
    global detector, detector_running
    try:
        settings = Settings()
        settings.load()
        
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
        print("🎤 Starting acoustic monitoring...")
        
        # Start FTP uploader if enabled
        settings_obj = Settings()
        settings_obj.load()
        if settings_obj.config.get('ftp', {}).get('enabled', False):
            ftp_uploader.start_uploader()
        
        detector.start_monitoring()
    except Exception as e:
        print(f"❌ Detector error: {e}")
        detector_running = False

# Web Routes
@app.route('/')
@login_required
def dashboard():
    recent_detections = detection_store.get_recent_detections(10)
    
    # Count only actual bird detections (all stored detections have birds)
    total_bird_detections = len(detection_store.detections)
    species_count = set()
    for detection in detection_store.detections:
        if 'analysis' in detection and 'species_detected' in detection['analysis']:
            species_count.update(detection['analysis']['species_detected'].keys())
    
    stats = {
        'total_detections': total_bird_detections,
        'unique_species': len(species_count),
        'detector_status': 'Running' if detector_running else 'Stopped',
        'last_detection': recent_detections[0]['formatted_time'] if recent_detections else 'None'
    }
    
    return render_template('dashboard.html', detections=recent_detections, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
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
    logout_user()
    return redirect(url_for('login'))

@app.route('/settings')
@login_required
def settings():
    settings = Settings()
    settings.load()
    return render_template('settings.html', settings=settings.config)

@app.route('/analyzer')
@login_required
def analyzer():
    return render_template('analyzer.html')

# API Routes
@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def api_settings():
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

@app.route('/api/analyze_audio', methods=['POST'])
@login_required
def analyze_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        audio_data, sample_rate = sf.read(temp_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        temp_detector = AcousticDetector()
        if temp_detector.bird_model is not None:
            analysis = temp_detector.analyze_recording_for_birds(audio_data)
        else:
            analysis = {'error': 'Bird detection model not available'}
        
        os.remove(temp_path)
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections')
@login_required
def api_detections():
    limit = request.args.get('limit', 20, type=int)
    detections = detection_store.get_recent_detections(limit)
    return jsonify(detections)

@app.route('/api/detector/start', methods=['POST'])
@login_required
def start_detector_api():
    global detector_thread, detector_running
    
    if not detector_running:
        detector_thread = threading.Thread(target=run_detector, daemon=True)
        detector_thread.start()
        return jsonify({'status': 'success', 'message': 'Detector started'})
    else:
        return jsonify({'status': 'info', 'message': 'Detector already running'})

@app.route('/api/detector/stop', methods=['POST'])
@login_required
def stop_detector_api():
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
    detection_store.clear_detections()
    return jsonify({'status': 'success', 'message': 'Detection history cleared'})

@app.route('/api/statistics')
@login_required
def api_statistics():
    detections = detection_store.detections
    
    species_freq = {}
    daily_counts = {}
    
    for detection in detections:
        if 'analysis' in detection and 'species_detected' in detection['analysis']:
            for species in detection['analysis']['species_detected'].keys():
                species_freq[species] = species_freq.get(species, 0) + 1
        
        date = detection['timestamp'][:10]
        daily_counts[date] = daily_counts.get(date, 0) + 1
    
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

@app.route('/api/audio/download/<filename>')
@login_required
def download_audio(filename):
    """Download audio file."""
    try:
        # Security check: ensure filename exists in our detections
        valid_file = False
        for detection in detection_store.detections:
            if detection.get('filename') == filename:
                valid_file = True
                break
        
        if not valid_file:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = os.path.join('recordings', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/stream/<filename>')
@login_required
def stream_audio(filename):
    """Stream audio file for playback."""
    try:
        # Security check: ensure filename exists in our detections
        valid_file = False
        for detection in detection_store.detections:
            if detection.get('filename') == filename:
                valid_file = True
                break
        
        if not valid_file:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = os.path.join('recordings', filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global detector, detector_running
    print("\n" + "="*60)
    print("⏹️  SHUTTING DOWN ACOUSTIC DETECTION SYSTEM")
    print("="*60)
    
    if detector and detector_running:
        print("🛑 Stopping detector...")
        detector_running = False
        detector.stop_monitoring()
    
    print("✅ System shutdown complete")
    print("="*60)
    sys.exit(0)

def main():
    """Main application entry point."""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
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
                'host': '0.0.0.0',  # Listen on all interfaces
                'port': 5000,
                'debug': False,
                'password': 'admin123'
            }
        }
        settings.save()
    
    # Get web configuration
    web_config = settings.config.get('web', {})
    host = web_config.get('host', '0.0.0.0')
    port = web_config.get('port', 5000)
    debug = web_config.get('debug', False)
    
    # Print startup information
    print_startup_info(host, port)
    
    # Start the detector automatically
    print("🚀 Auto-starting detector...")
    detector_thread = threading.Thread(target=run_detector, daemon=True)
    detector_thread.start()
    
    # Small delay to let detector initialize
    time.sleep(2)
    
    try:
        # Start Flask app
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
