import cv2
import numpy as np
from typing import Tuple, Optional, List

# Import MediaPipe with compatibility handling
try:
    import mediapipe as mp
    # Try to access solutions API (old API)
    try:
        _mp_face_mesh = mp.solutions.face_mesh
        _mp_drawing = mp.solutions.drawing_utils
        HAS_OLD_API = True
    except AttributeError:
        HAS_OLD_API = False
        _mp_face_mesh = None
        _mp_drawing = None
    
    # Try to access tasks API (new API)
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        HAS_NEW_API = True
    except (ImportError, AttributeError):
        HAS_NEW_API = False
        mp_tasks = None
        vision = None
        
except ImportError:
    raise ImportError("MediaPipe not installed. Please install it with: pip install mediapipe")

if not HAS_OLD_API and not HAS_NEW_API:
    raise ImportError("MediaPipe API not available. Please reinstall mediapipe.")

class EyeDetector:
    def __init__(self):
        self.use_new_api = not HAS_OLD_API
        
        if HAS_OLD_API:
            # Use old solutions API
            self.mp_face_mesh = _mp_face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = _mp_drawing
        elif HAS_NEW_API:
            # For new API, we'll use face_recognition as fallback since new API requires model files
            # Try to use face_recognition library instead
            try:
                import face_recognition
                self.use_face_recognition = True
                self.face_landmarker = None
            except ImportError:
                raise ImportError("New MediaPipe API requires model files. Please install face_recognition: pip install face_recognition")
            self.mp_drawing = None
        else:
            # Fallback to face_recognition if MediaPipe not available
            try:
                import face_recognition
                self.use_face_recognition = True
                self.use_new_api = False
            except ImportError:
                raise ImportError("No compatible face detection library found. Please install mediapipe or face_recognition.")
        
        # Eye landmark indices (MediaPipe Face Mesh)
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Key points for eye aspect ratio calculation
        self.LEFT_EYE_POINTS = [33, 160, 158, 153, 133, 157]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 373, 263, 386]
    
    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """Detect face and return landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.use_new_api and hasattr(self, 'face_mesh'):
            # Old MediaPipe API
            results = self.face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                return {
                    'landmarks': face_landmarks,
                    'image_shape': image.shape
                }
        elif hasattr(self, 'use_face_recognition') and self.use_face_recognition:
            # Fallback to face_recognition - create dummy landmarks structure
            import face_recognition
            face_locations = face_recognition.face_locations(rgb_image, model='hog')
            if len(face_locations) > 0:
                # Create a simple landmark structure for compatibility
                # Note: This is a simplified approach - real landmarks would need MediaPipe
                class SimpleLandmark:
                    def __init__(self, x, y, z=0):
                        self.x = x
                        self.y = y
                        self.z = z
                
                class SimpleLandmarks:
                    def __init__(self, face_location, image_shape):
                        top, right, bottom, left = face_location
                        h, w = image_shape[:2]
                        # Create approximate eye landmarks based on face location
                        # This is a simplified approximation
                        center_x = (left + right) / 2.0 / w
                        center_y = (top + bottom) / 2.0 / h
                        eye_y = (top + (bottom - top) * 0.4) / h
                        
                        # Create minimal landmark set for eye detection
                        self.landmark = [SimpleLandmark(0, 0) for _ in range(468)]
                        # Set approximate eye positions
                        # Left eye center (approximate)
                        self.landmark[33] = SimpleLandmark(center_x - 0.1, eye_y)
                        self.landmark[160] = SimpleLandmark(center_x - 0.15, eye_y)
                        self.landmark[158] = SimpleLandmark(center_x - 0.1, eye_y - 0.02)
                        self.landmark[153] = SimpleLandmark(center_x - 0.05, eye_y)
                        self.landmark[133] = SimpleLandmark(center_x - 0.1, eye_y + 0.02)
                        self.landmark[157] = SimpleLandmark(center_x - 0.15, eye_y)
                        # Right eye center (approximate)
                        self.landmark[362] = SimpleLandmark(center_x + 0.1, eye_y)
                        self.landmark[385] = SimpleLandmark(center_x + 0.15, eye_y)
                        self.landmark[387] = SimpleLandmark(center_x + 0.1, eye_y - 0.02)
                        self.landmark[373] = SimpleLandmark(center_x + 0.05, eye_y)
                        self.landmark[263] = SimpleLandmark(center_x + 0.1, eye_y + 0.02)
                        self.landmark[386] = SimpleLandmark(center_x + 0.15, eye_y)
                
                return {
                    'landmarks': SimpleLandmarks(face_locations[0], image.shape),
                    'image_shape': image.shape
                }
        
        return None
    
    def get_eye_region(self, image: np.ndarray, landmarks, eye_indices: List[int]) -> Optional[np.ndarray]:
        """Extract eye region from face landmarks"""
        h, w = image.shape[:2]
        
        # Get eye landmark coordinates
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            eye_points.append([x, y])
        
        eye_points = np.array(eye_points)
        
        # Get bounding box with padding
        x_min = max(0, int(eye_points[:, 0].min()) - 10)
        x_max = min(w, int(eye_points[:, 0].max()) + 10)
        y_min = max(0, int(eye_points[:, 1].min()) - 10)
        y_max = min(h, int(eye_points[:, 1].max()) + 10)
        
        # Crop eye region
        eye_region = image[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        return eye_region
    
    def get_both_eyes_region(self, image: np.ndarray, landmarks) -> Optional[np.ndarray]:
        """Get combined region containing both eyes"""
        h, w = image.shape[:2]
        
        # Get left eye points
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            left_eye_points.append([x, y])
        
        # Get right eye points
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            right_eye_points.append([x, y])
        
        all_points = np.array(left_eye_points + right_eye_points)
        
        # Get bounding box with padding
        x_min = max(0, int(all_points[:, 0].min()) - 20)
        x_max = min(w, int(all_points[:, 0].max()) + 20)
        y_min = max(0, int(all_points[:, 1].min()) - 20)
        y_max = min(h, int(all_points[:, 1].max()) + 20)
        
        # Crop eye region
        eye_region = image[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        return eye_region
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_points: List[int], image_shape: Tuple[int, int] = None) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # MediaPipe landmarks are normalized (0-1), so we don't need image dimensions
        # But we'll use them if provided for consistency
        
        # Get vertical distances (using normalized coordinates)
        vertical_1 = np.linalg.norm([
            landmarks.landmark[eye_points[1]].x - landmarks.landmark[eye_points[5]].x,
            landmarks.landmark[eye_points[1]].y - landmarks.landmark[eye_points[5]].y
        ])
        vertical_2 = np.linalg.norm([
            landmarks.landmark[eye_points[2]].x - landmarks.landmark[eye_points[4]].x,
            landmarks.landmark[eye_points[2]].y - landmarks.landmark[eye_points[4]].y
        ])
        
        # Get horizontal distance
        horizontal = np.linalg.norm([
            landmarks.landmark[eye_points[0]].x - landmarks.landmark[eye_points[3]].x,
            landmarks.landmark[eye_points[0]].y - landmarks.landmark[eye_points[3]].y
        ])
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def detect_blink(self, landmarks, image_shape: Tuple[int, int] = None) -> Tuple[float, float, float]:
        """Detect blink by calculating EAR for both eyes"""
        left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_POINTS, image_shape)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_POINTS, image_shape)
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear, left_ear, right_ear
    
    def preprocess_eye_region(self, eye_region: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """Preprocess eye region for embedding extraction"""
        # Resize
        resized = cv2.resize(eye_region, target_size)
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Normalize
        normalized = equalized.astype(np.float32) / 255.0
        
        return normalized
    
    def draw_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Draw face landmarks on image for visualization"""
        if not self.use_new_api and self.mp_drawing:
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                None,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            return annotated_image
        else:
            # For new API, return image as-is (drawing not implemented)
            return image.copy()
