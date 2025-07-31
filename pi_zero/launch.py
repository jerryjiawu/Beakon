#!/usr/bin/env python3
"""
Acoustic Detection System - Raspberry Pi Zero W Optimized Version
Lightweight version optimized for low-resource devices.
"""

import sys
import os
import socket
import threading
import time
import signal
import ftplib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import json
import datetime
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.detector import AcousticDetector
from src.settings import Settings

# Global variables
detector = None
detector_thread = None
detector_running = False

def get_local_ip():
    """Get the local IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def print_startup_info(host, port):
    """Print startup information."""
    local_ip = get_local_ip()
    print("\n" + "="*50)
    print("🎤 ACOUSTIC DETECTION (Pi Zero W)")
    print("="*50)
    print(f"🌐 Local:    http://localhost:{port}")
    print(f"🌍 Network:  http://{local_ip}:{port}")
    print(f"🔑 Password: admin123")
    print("="*50)
    print("⏹️  Press Ctrl+C to stop")
    print("="*50)

# Lightweight Flask App
app = Flask(__name__, template_folder='templates')
app.secret_key = 'pi-zero-acoustic-detection'

# Simplified login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

def get_system_password():
    """Get password from settings."""
    settings = Settings()
    settings.load()
    return settings.config.get('web', {}).get('password', 'admin123')

@login_manager.user_loader
def load_user(user_id):
    if user_id == "user":
        return User(user_id)
    return None

class LightweightDetectionStore:
    """Memory-efficient detection storage."""
    def __init__(self):
        self.detections = []
        self.max_detections = 5  # Reduced for Pi Zero W
    
    def add_detection(self, detection_data):
        # Only store detections with birds
        if detection_data.get('analysis', {}).get('unique_species', 0) > 0:
            # Keep only essential data to save memory
            essential_data = {
                'timestamp': detection_data['timestamp'],
                'formatted_time': detection_data['formatted_time'],
                'duration': detection_data['duration'],
                'species_count': detection_data['analysis']['unique_species'],
                'top_species': self._get_top_species(detection_data['analysis'])
            }
            
            # Add audio file if it exists
            if 'filename' in detection_data:
                essential_data['filename'] = detection_data['filename']
                essential_data['audio_file'] = detection_data['audio_file']
            
            self.detections.insert(0, essential_data)
            
            # Clean up old detections
            if len(self.detections) > self.max_detections:
                for old_detection in self.detections[self.max_detections:]:
                    if 'audio_file' in old_detection:
                        try:
                            os.remove(old_detection['audio_file'])
                        except:
                            pass
                self.detections = self.detections[:self.max_detections]
    
    def _get_top_species(self, analysis):
        """Extract top 3 species to save memory."""
        if 'species_detected' not in analysis:
            return []
        
        sorted_species = sorted(
            analysis['species_detected'].items(),
            key=lambda x: x[1]['max_confidence'],
            reverse=True
        )
        
        return [{
            'name': species,
            'confidence': data['max_confidence']
        } for species, data in sorted_species[:3]]
    
    def get_recent_detections(self, limit=5):
        return self.detections[:min(limit, len(self.detections))]
    
    def clear_detections(self):
        for detection in self.detections:
            if 'audio_file' in detection:
                try:
                    os.remove(detection['audio_file'])
                except:
                    pass
        self.detections = []

detection_store = LightweightDetectionStore()

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
            
            recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
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

# Lightweight detector for Pi Zero W
class PiZeroDetector(AcousticDetector):
    """Pi Zero W optimized detector."""
    
    def __init__(self, **kwargs):
        # Override settings for Pi Zero W optimization
        pi_settings = {
            'sample_rate': 22050,  # Reduced from 44100
            'chunk_size': 512,     # Reduced from 1024
            'recording_duration': 8.0,  # Reduced from 10.0
            'baseline_window': 3.0,     # Reduced from 5.0
            'min_bird_confidence': 0.7  # Increased threshold
        }
        pi_settings.update(kwargs)
        super().__init__(**pi_settings)
    
    def process_recording(self):
        if self.recording_data:
            timestamp = datetime.datetime.now()
            full_recording = np.concatenate(self.recording_data)
            duration = len(full_recording) / self.sample_rate
            
            print(f"🔍 Analyzing {duration:.1f}s...")
            
            # Quick analysis with reduced processing
            bird_analysis = self.analyze_recording_for_birds(
                full_recording,
                sensitivity=0.8,  # Lower sensitivity for faster processing
                min_confidence=self.min_bird_confidence
            )
            
            # Only save if birds detected
            if not bird_analysis.get('error') and bird_analysis.get('unique_species', 0) > 0:
                recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
                os.makedirs(recordings_dir, exist_ok=True)
                filename = f"bird_{timestamp.strftime('%H%M%S')}.wav"
                audio_file_path = os.path.join(recordings_dir, filename)
                
                try:
                    # Save as 16-bit to reduce file size
                    import soundfile as sf
                    # Normalize and convert to int16 to save space
                    normalized_audio = np.clip(full_recording * 32767, -32768, 32767).astype(np.int16)
                    sf.write(audio_file_path, normalized_audio, self.sample_rate, subtype='PCM_16')
                    
                    detection_data = {
                        'timestamp': timestamp.isoformat(),
                        'duration': duration,
                        'analysis': bird_analysis,
                        'formatted_time': timestamp.strftime("%H:%M:%S"),
                        'audio_file': audio_file_path,
                        'filename': filename
                    }
                    detection_store.add_detection(detection_data)
                    
                    # Queue for FTP upload
                    ftp_uploader.queue_upload(audio_file_path, filename)
                    
                    print(f"🐦 {bird_analysis['unique_species']} species → {filename}")
                    
                except Exception as e:
                    print(f"❌ Save error: {e}")
            else:
                print("🔍 No birds detected")
        
        # Clean up
        self.is_recording = False
        self.recording_start_time = None
        self.recording_data = []

def run_detector():
    """Run optimized detector for Pi Zero W."""
    global detector, detector_running
    try:
        settings = Settings()
        settings.load()
        
        detector_config = settings.config.get('detector', {})
        detector = PiZeroDetector(
            sample_rate=detector_config.get('sample_rate', 22050),
            chunk_size=detector_config.get('chunk_size', 512),
            low_freq=detector_config.get('low_freq', 1500),
            high_freq=detector_config.get('high_freq', 6000),
            spike_threshold=detector_config.get('spike_threshold', 2.5),
            recording_duration=detector_config.get('recording_duration', 8.0),
            baseline_window=detector_config.get('baseline_window', 3.0),
            min_bird_confidence=detector_config.get('min_bird_confidence', 0.7)
        )
        
        detector_running = True
        print("🎤 Starting Pi Zero monitoring...")
        
        # Start FTP uploader if enabled
        settings_obj = Settings()
        settings_obj.load()
        if settings_obj.config.get('ftp', {}).get('enabled', False):
            ftp_uploader.start_uploader()
        
        detector.start_monitoring()
    except Exception as e:
        print(f"❌ Detector error: {e}")
        detector_running = False

# Simplified routes
@app.route('/')
@login_required
def dashboard():
    recent_detections = detection_store.get_recent_detections(5)
    
    stats = {
        'total_detections': len(detection_store.detections),
        'detector_status': 'Running' if detector_running else 'Stopped',
        'last_detection': recent_detections[0]['formatted_time'] if recent_detections else 'None'
    }
    
    return render_template('pi_dashboard.html', detections=recent_detections, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form['password']
        if password == get_system_password():
            user = User("user")
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid password')
    return render_template('pi_login.html')

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
    return render_template('pi_settings.html', settings=settings.config)

# API routes
@app.route('/api/detections')
@login_required
def api_detections():
    detections = detection_store.get_recent_detections(5)
    return jsonify(detections)

@app.route('/api/detector/start', methods=['POST'])
@login_required
def start_detector():
    global detector_thread, detector_running
    if not detector_running:
        detector_thread = threading.Thread(target=run_detector, daemon=True)
        detector_thread.start()
        return jsonify({'status': 'success', 'message': 'Started'})
    return jsonify({'status': 'info', 'message': 'Already running'})

@app.route('/api/detector/stop', methods=['POST'])
@login_required
def stop_detector():
    global detector, detector_running
    if detector and detector_running:
        detector_running = False
        detector.stop_monitoring()
        return jsonify({'status': 'success', 'message': 'Stopped'})
    return jsonify({'status': 'info', 'message': 'Not running'})

@app.route('/api/detections/clear', methods=['POST'])
@login_required
def clear_detections():
    detection_store.clear_detections()
    return jsonify({'status': 'success', 'message': 'Cleared'})

@app.route('/api/settings', methods=['GET', 'POST'])
@login_required
def api_settings():
    settings = Settings()
    if request.method == 'GET':
        settings.load()
        return jsonify(settings.config)
    else:
        try:
            settings.config = request.json
            settings.save()
            return jsonify({'status': 'success', 'message': 'Saved'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/audio/download/<filename>')
@login_required
def download_audio(filename):
    """Download audio file."""
    try:
        # Security check
        valid_file = any(d.get('filename') == filename for d in detection_store.detections)
        if not valid_file:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = os.path.join(os.path.dirname(__file__), 'recordings', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/stream/<filename>')
@login_required
def stream_audio(filename):
    """Stream audio for playback."""
    try:
        # Security check
        valid_file = any(d.get('filename') == filename for d in detection_store.detections)
        if not valid_file:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = os.path.join(os.path.dirname(__file__), 'recordings', filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def signal_handler(sig, frame):
    """Handle shutdown."""
    global detector, detector_running
    print("\n" + "="*50)
    print("⏹️  SHUTTING DOWN")
    print("="*50)
    
    if detector and detector_running:
        detector_running = False
        detector.stop_monitoring()
    
    print("✅ Stopped")
    sys.exit(0)

def main():
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create Pi Zero W optimized settings
    settings = Settings()
    settings.load()
    if not settings.config:
        settings.config = {
            'detector': {
                'sample_rate': 22050,      # Reduced for Pi Zero W
                'chunk_size': 512,         # Reduced for Pi Zero W
                'channels': 1,
                'low_freq': 1500,          # Narrower range
                'high_freq': 6000,         # Narrower range
                'spike_threshold': 2.5,    # Higher threshold
                'recording_duration': 8.0, # Shorter recordings
                'baseline_window': 3.0,    # Shorter baseline
                'min_bird_confidence': 0.7 # Higher confidence
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'password': 'admin123'
            }
        }
        settings.save()
    
    web_config = settings.config.get('web', {})
    host = web_config.get('host', '0.0.0.0')
    port = web_config.get('port', 5000)
    
    print_startup_info(host, port)
    
    # Auto-start detector
    print("🚀 Starting detector...")
    detector_thread = threading.Thread(target=run_detector, daemon=True)
    detector_thread.start()
    
    time.sleep(1)
    
    try:
        # Run with minimal threads for Pi Zero W
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
