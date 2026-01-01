import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from eye_detection import EyeDetector
from config import Config

class LivenessDetector:
    def __init__(self):
        self.config = Config()
        self.eye_detector = EyeDetector()
        self.ear_threshold = self.config.EYE_ASPECT_RATIO_THRESHOLD
        self.consecutive_frames = self.config.EYE_AR_CONSEC_FRAMES
        self.session_timeout = getattr(self.config, "LIVENESS_SESSION_TIMEOUT", 45)
        self._sessions: Dict[str, dict] = {}
        self._default_session_key = "__default__"
        self._sessions[self._default_session_key] = self._create_session_state()
    
    def _create_session_state(self) -> dict:
        return {
            'blink_counter': 0,
            'frame_counter': 0,
            'ear_history': [],
            'last_updated': datetime.utcnow()
        }
    
    def _cleanup_sessions(self):
        """Remove stale session states"""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, state in self._sessions.items()
            if session_id != self._default_session_key and
            (now - state['last_updated']) > timedelta(seconds=self.session_timeout)
        ]
        for session_id in expired_sessions:
            self._sessions.pop(session_id, None)
    
    def _get_session_state(self, session_id: Optional[str]) -> dict:
        """Get or create session-specific state"""
        self._cleanup_sessions()
        
        if not session_id:
            session_id = self._default_session_key
        
        if session_id not in self._sessions:
            self._sessions[session_id] = self._create_session_state()
        
        self._sessions[session_id]['last_updated'] = datetime.utcnow()
        return self._sessions[session_id]
    
    def reset(self, session_id: Optional[str] = None):
        """Reset liveness detection state"""
        if session_id:
            self._sessions[session_id] = self._create_session_state()
        else:
            self._sessions = {
                self._default_session_key: self._create_session_state()
            }
    
    def detect_blink_sequence(self, image: np.ndarray, session_id: Optional[str] = None) -> Tuple[bool, float, dict]:
        """
        Detect blink sequence for liveness verification
        Returns: (is_live, current_ear, status_info)
        """
        state = self._get_session_state(session_id)
        face_data = self.eye_detector.detect_face(image)
        
        if face_data is None:
            return False, 0.0, {'status': 'no_face', 'message': 'No face detected'}
        
        landmarks = face_data['landmarks']
        image_shape = face_data.get('image_shape', image.shape[:2])
        avg_ear, left_ear, right_ear = self.eye_detector.detect_blink(landmarks, image_shape)
        
        state['frame_counter'] += 1
        state['ear_history'].append(avg_ear)
        
        # Keep only last 30 frames
        if len(state['ear_history']) > 30:
            state['ear_history'].pop(0)
        
        # Detect blink (EAR drops below threshold)
        if avg_ear < self.ear_threshold:
            state['blink_counter'] += 1
        else:
            # If EAR is above threshold and we had a blink, reset counter
            if state['blink_counter'] >= self.consecutive_frames:
                # Blink detected
                frames_analyzed = state['frame_counter']
                state['blink_counter'] = 0
                state['frame_counter'] = 0
                state['ear_history'] = []
                return True, avg_ear, {
                    'status': 'blink_detected',
                    'message': 'Blink detected - liveness confirmed',
                    'ear': avg_ear,
                    'frames_analyzed': frames_analyzed
                }
            state['blink_counter'] = 0
        
        # Check if we've analyzed enough frames
        if state['frame_counter'] < 10:
            return False, avg_ear, {
                'status': 'analyzing',
                'message': f'Analyzing... ({state["frame_counter"]}/10 frames)',
                'ear': avg_ear,
                'frames_analyzed': state['frame_counter']
            }
        
        # If we've analyzed many frames without detecting a blink, might be a photo
        if state['frame_counter'] > 60:
            self.reset(session_id)
            return False, avg_ear, {
                'status': 'no_blink',
                'message': 'No blink detected - possible spoofing attempt',
                'ear': avg_ear,
                'frames_analyzed': state['frame_counter']
            }
        
        return False, avg_ear, {
            'status': 'waiting',
            'message': 'Please blink naturally',
            'ear': avg_ear,
            'frames_analyzed': state['frame_counter']
        }
    
    def quick_liveness_check(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Quick liveness check - analyzes single frame for basic signs of liveness
        Returns: (is_live, status_info)
        """
        face_data = self.eye_detector.detect_face(image)
        
        if face_data is None:
            return False, {'status': 'no_face', 'message': 'No face detected'}
        
        landmarks = face_data['landmarks']
        image_shape = face_data.get('image_shape', image.shape[:2])
        avg_ear, left_ear, right_ear = self.eye_detector.detect_blink(landmarks, image_shape)
        
        # Basic checks
        # 1. EAR should be in reasonable range (not too low, not too high)
        if avg_ear < 0.15 or avg_ear > 0.5:
            return False, {
                'status': 'suspicious',
                'message': 'Unusual eye aspect ratio detected',
                'ear': avg_ear
            }
        
        # 2. Both eyes should have similar EAR (symmetry check)
        ear_diff = abs(left_ear - right_ear)
        if ear_diff > 0.1:
            return False, {
                'status': 'suspicious',
                'message': 'Asymmetric eye measurements detected',
                'ear': avg_ear,
                'ear_diff': ear_diff
            }
        
        return True, {
            'status': 'passed',
            'message': 'Basic liveness check passed',
            'ear': avg_ear,
            'left_ear': left_ear,
            'right_ear': right_ear
        }
    
    def verify_liveness_with_challenge(self, frames: List[np.ndarray]) -> Tuple[bool, dict]:
        """
        Verify liveness using multiple frames and challenge-response
        Analyzes frame sequence for natural movement
        """
        if len(frames) < 5:
            return False, {'status': 'insufficient_frames', 'message': 'Need at least 5 frames'}
        
        ear_values = []
        face_detected_count = 0
        
        for frame in frames:
            face_data = self.eye_detector.detect_face(frame)
            if face_data:
                face_detected_count += 1
                landmarks = face_data['landmarks']
                image_shape = face_data.get('image_shape', frame.shape[:2])
                avg_ear, _, _ = self.eye_detector.detect_blink(landmarks, image_shape)
                ear_values.append(avg_ear)
        
        if face_detected_count < len(frames) * 0.8:
            return False, {
                'status': 'inconsistent',
                'message': 'Face not consistently detected across frames'
            }
        
        if len(ear_values) < 3:
            return False, {
                'status': 'insufficient_data',
                'message': 'Insufficient data for liveness verification'
            }
        
        # Check for variation in EAR (indicates natural movement)
        ear_std = np.std(ear_values)
        ear_mean = np.mean(ear_values)
        
        # Natural faces should have some variation
        if ear_std < 0.01:
            return False, {
                'status': 'static',
                'message': 'No natural movement detected - possible photo/video',
                'ear_std': ear_std,
                'ear_mean': ear_mean
            }
        
        # Check for reasonable EAR range
        if ear_mean < 0.15 or ear_mean > 0.5:
            return False, {
                'status': 'unusual',
                'message': 'Unusual eye measurements',
                'ear_mean': ear_mean
            }
        
        return True, {
            'status': 'passed',
            'message': 'Liveness verified through frame analysis',
            'ear_mean': ear_mean,
            'ear_std': ear_std,
            'frames_analyzed': len(ear_values)
        }

