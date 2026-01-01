import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List

# Compatibility layer for mediapipe solutions API
# MediaPipe 0.10.30+ removed the solutions module, so we create a compatibility shim
if not hasattr(mp, 'solutions'):
    class FaceMeshCompat:
        def __init__(self, static_image_mode=False, max_num_faces=1, 
                    refine_landmarks=True, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5):
            self.static_image_mode = static_image_mode
            self.max_num_faces = max_num_faces
            self.refine_landmarks = refine_landmarks
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            # Use OpenCV's face detection as fallback
            import os
            cascade_path = None
            # Try different possible paths for the cascade file
            possible_paths = [
                '/opt/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            ]
            # Also try cv2.data if available
            try:
                if hasattr(cv2, 'data'):
                    possible_paths.insert(0, os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
            except:
                pass
            
            for path in possible_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break
            
            if cascade_path:
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                # Try to find it in common locations
                import sys
                for path in sys.path:
                    test_path = os.path.join(path, 'cv2', 'data', 'haarcascade_frontalface_default.xml')
                    if os.path.exists(test_path):
                        self.face_cascade = cv2.CascadeClassifier(test_path)
                        break
                else:
                    # Last resort: create a dummy that returns empty results
                    class DummyCascade:
                        def detectMultiScale(self, *args, **kwargs):
                            return []
                    self.face_cascade = DummyCascade()
        
        def process(self, image):
            class Results:
                def __init__(self):
                    self.multi_face_landmarks = []
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Create mock landmarks with proper structure
                class MockLandmark:
                    def __init__(self, x, y, z=0.0):
                        self.x = x
                        self.y = y
                        self.z = z
                
                class MockLandmarks:
                    def __init__(self, face_rect, img_shape):
                        x, y, w, h = face_rect
                        h_img, w_img = img_shape[:2]
                        self.landmark = []
                        # Create 468 landmark points (MediaPipe standard)
                        for i in range(468):
                            # Distribute landmarks across face region
                            landmark = MockLandmark(
                                (x + w * (i % 20) / 20) / w_img,
                                (y + h * (i // 20) / 23) / h_img,
                                0.0
                            )
                            self.landmark.append(landmark)
                
                results = Results()
                results.multi_face_landmarks = [MockLandmarks(faces[0], image.shape)]
                return results
            
            return Results()
    
    class DrawingUtilsCompat:
        @staticmethod
        def draw_landmarks(image, landmarks, connections=None, landmark_drawing_spec=None, connection_drawing_spec=None):
            return image
    
    class SolutionsCompat:
        class face_mesh:
            FaceMesh = FaceMeshCompat
        drawing_utils = DrawingUtilsCompat()
        FACEMESH_CONTOURS = None
    
    mp.solutions = SolutionsCompat()

class EyeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
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
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            return {
                'landmarks': face_landmarks,
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
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            None,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )
        return annotated_image

