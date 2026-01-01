from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
from datetime import datetime, date, time
import os
from pathlib import Path
from database import Database
from recognition import RecognitionSystem
from liveness_detection import LivenessDetector
from eye_detection import EyeDetector
from config import Config

# Get the project root directory (parent of backend)
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

app = Flask(__name__, 
            template_folder=str(FRONTEND_DIR / 'templates'), 
            static_folder=str(FRONTEND_DIR / 'static'))
CORS(app)
app.config['SECRET_KEY'] = Config().SECRET_KEY

# Initialize components
db = Database()
recognition = RecognitionSystem()
liveness_detector = LivenessDetector()
eye_detector = EyeDetector()

# Create uploads directory in backend folder
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

def decode_image(base64_string):
    """Decode base64 image string to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Serve admin page"""
    return render_template('admin.html')

@app.route('/verify')
def verify_page():
    """Serve verification page"""
    return render_template('verify.html')

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register a new user with enrollment samples"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        notes = data.get('notes')
        images = data.get('images', [])  # List of base64 encoded images
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        if len(images) == 0:
            return jsonify({'error': 'At least one image is required'}), 400
        
        # Create user in database
        user_id = db.insert_user(name, email, notes)
        
        # Process and store embeddings
        embeddings = []
        for idx, img_base64 in enumerate(images):
            image = decode_image(img_base64)
            if image is None:
                continue
            
            # Extract embedding
            embedding = recognition.extract_embedding_mediapipe(image)
            if embedding is None:
                continue
            
            # Save sample image
            sample_path = str(UPLOAD_FOLDER / f'user_{user_id}_sample_{idx}.jpg')
            cv2.imwrite(sample_path, image)
            
            # Store embedding
            db.insert_embedding(user_id, embedding, sample_path)
            embeddings.append(embedding)
        
        if len(embeddings) == 0:
            # Delete user if no valid embeddings
            db.delete_user(user_id)
            return jsonify({'error': 'Could not extract valid embeddings from images'}), 400
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'name': name,
            'samples_collected': len(embeddings),
            'message': 'User registered successfully'
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_user():
    """Verify user identity and mark attendance"""
    try:
        data = request.json
        image_base64 = data.get('image')
        device_id = data.get('device_id', 'webcam')
        session_id = data.get('session_id')
        
        if not image_base64:
            return jsonify({'error': 'Image is required'}), 400
        
        # Decode image
        image = decode_image(image_base64)
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Extract embedding
        embedding = recognition.extract_embedding_mediapipe(image)
        if embedding is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in image',
                'user_id': None,
                'confidence': 0.0,
                'liveness': 'failed'
            }), 200
        
        # Liveness check (session-aware blink detection if session id provided)
        if session_id:
            is_live, _, liveness_info = liveness_detector.detect_blink_sequence(
                image, session_id=session_id
            )
        else:
            is_live, liveness_info = liveness_detector.quick_liveness_check(image)

        liveness_status = liveness_info.get('status') if liveness_info else None

        # Hard‑fail only on clear liveness problems; allow verification to proceed
        # while still surfacing liveness status to the client.
        hard_fail_statuses = {'no_face', 'no_blink', 'suspicious', 'static', 'unusual'}
        if not is_live and liveness_status in hard_fail_statuses:
            error_message = liveness_info.get('message', 'Liveness check failed')
            if liveness_status == 'no_face':
                error_message = 'No face detected - please adjust your position'
            elif liveness_status == 'no_blink':
                error_message = 'Blink not detected - try blinking naturally'

            return jsonify({
                'success': False,
                'error': error_message,
                'stage': 'liveness',
                'liveness': liveness_status or 'failed',
                'liveness_info': liveness_info,
                'user_id': None,
                'confidence': 0.0,
                'session_id': session_id
            }), 200
        
        # Get all users with embeddings
        users = db.get_all_users_with_embeddings()
        
        if len(users) == 0:
            return jsonify({
                'success': False,
                'error': 'No registered users found',
                'user_id': None,
                'confidence': 0.0,
                'liveness': 'passed'
            }), 200
        
        # Match against all users
        best_match = None
        best_confidence = 0.0
        best_user_id = None
        
        for user in users:
            user_id = user['user_id']
            stored_embeddings = db.get_user_embeddings(user_id)
            
            if len(stored_embeddings) == 0:
                continue
            
            # Calculate average embedding for user
            avg_embedding = recognition.average_embeddings(stored_embeddings)
            
            # Match
            similarity = recognition.calculate_similarity(embedding, avg_embedding)
            
            if similarity > best_confidence:
                best_confidence = similarity
                best_user_id = user_id
                best_match = user
        
        # Check if match meets threshold
        if best_confidence < recognition.similarity_threshold:
            return jsonify({
                'success': False,
                'error': 'No matching user found',
                'user_id': None,
                'confidence': best_confidence,
                'liveness': 'passed'
            }), 200
        
        # Mark attendance
        today = date.today()
        now = datetime.now()
        current_time = now.time()
        
        attendance_id = db.mark_attendance(
            user_id=best_user_id,
            date=today,
            time=current_time,
            device_id=device_id,
            status='Present',
            confidence=best_confidence,
            liveness_status=liveness_status or 'passed',
            notes=None
        )
        
        if session_id:
            liveness_detector.reset(session_id)
        
        return jsonify({
            'success': True,
            'user_id': best_user_id,
            'name': best_match['name'],
            'email': best_match.get('email'),
            'confidence': float(best_confidence),
            'liveness': 'passed',
            'liveness_info': liveness_info,
            'attendance_marked': True,
            'attendance_id': attendance_id,
            'timestamp': now.isoformat(),
            'session_id': session_id
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all registered users"""
    try:
        users = db.get_all_users()
        return jsonify({'users': users}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    try:
        db.delete_user(user_id)
        return jsonify({'success': True, 'message': 'User deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    """Get attendance records"""
    try:
        date_str = request.args.get('date')
        user_id = request.args.get('user_id', type=int)
        
        if date_str:
            # Get attendance for specific date
            records = db.get_attendance_by_date(date_str)
        elif user_id:
            # Get attendance for specific user
            records = db.get_user_attendance(user_id)
        else:
            # Get today's attendance
            today = date.today().isoformat()
            records = db.get_attendance_by_date(today)
        
        return jsonify({'attendance': records}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/liveness/check', methods=['POST'])
def check_liveness():
    """Check liveness from image sequence"""
    try:
        data = request.json
        images = data.get('images', [])  # List of base64 images
        
        if len(images) == 0:
            return jsonify({'error': 'At least one image is required'}), 400
        
        # Decode images
        frames = []
        for img_base64 in images:
            image = decode_image(img_base64)
            if image is not None:
                frames.append(image)
        
        if len(frames) == 0:
            return jsonify({'error': 'No valid images provided'}), 400
        
        # Verify liveness
        is_live, info = liveness_detector.verify_liveness_with_challenge(frames)
        
        return jsonify({
            'liveness_passed': is_live,
            'info': info
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        users = db.get_all_users()
        today = date.today().isoformat()
        today_attendance = db.get_attendance_by_date(today)
        
        return jsonify({
            'total_users': len(users),
            'today_attendance': len(today_attendance),
            'date': today
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create database tables on startup
    try:
        db.create_tables()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Note: Make sure MySQL is running and database 'attendance_system' exists")
        print("You can create it with: CREATE DATABASE attendance_system;")
    
    port = Config().PORT
    # Use port 5001 if 5000 is unavailable
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(('localhost', 5000)) == 0:
        port = 5001
        print(f"Port 5000 is in use, using port {port} instead")
    sock.close()
    
    print(f"\n{'='*50}")
    print(f"Smart Attendance System - Eye Scan")
    print(f"{'='*50}")
    print(f"Server starting on http://localhost:{port}")
    print(f"Press Ctrl+C to stop the server")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=port, debug=Config().DEBUG)

