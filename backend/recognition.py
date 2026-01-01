import face_recognition
import numpy as np
import cv2
from typing import List, Tuple, Optional
from eye_detection import EyeDetector
from config import Config

class RecognitionSystem:
    def __init__(self):
        self.config = Config()
        self.eye_detector = EyeDetector()
        self.similarity_threshold = self.config.SIMILARITY_THRESHOLD
    
    def extract_embedding_from_eye_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using eye region approach"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if len(face_locations) == 0:
            return None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return None
        
        # Return the first face encoding (128-dimensional vector)
        return face_encodings[0]
    
    def extract_embedding_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using MediaPipe face detection and face_recognition"""
        # Detect face using MediaPipe
        face_data = self.eye_detector.detect_face(image)
        
        if face_data is None:
            return None
        
        # Convert to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face bounding box from landmarks
        landmarks = face_data['landmarks']
        h, w = image.shape[:2]
        
        # Calculate bounding box from all landmarks
        x_coords = [landmark.x * w for landmark in landmarks.landmark]
        y_coords = [landmark.y * h for landmark in landmarks.landmark]
        
        top = int(min(y_coords))
        right = int(max(x_coords))
        bottom = int(max(y_coords))
        left = int(min(x_coords))
        
        # Expand bounding box slightly
        padding = 20
        top = max(0, top - padding)
        right = min(w, right + padding)
        bottom = min(h, bottom + padding)
        left = max(0, left - padding)
        
        face_location = (top, right, bottom, left)
        
        face_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=[face_location],
            num_jitters=1
        )
        
        if len(face_encodings) == 0:
            face_encodings = face_recognition.face_encodings(rgb_image)
        
        if len(face_encodings) == 0:
            return None
        
        return face_encodings[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def match_embedding(self, query_embedding: np.ndarray, 
                       stored_embeddings: List[np.ndarray]) -> Tuple[Optional[int], float]:
        """Match query embedding against stored embeddings"""
        if len(stored_embeddings) == 0:
            return None, 0.0
        
        best_match_idx = None
        best_similarity = 0.0
        
        for idx, stored_embedding in enumerate(stored_embeddings):
            similarity = self.calculate_similarity(query_embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_idx, best_similarity
        
        return None, best_similarity
    
    def average_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate average embedding from multiple samples"""
        if len(embeddings) == 0:
            return None
        
        # Stack embeddings and calculate mean
        stacked = np.stack(embeddings)
        avg_embedding = np.mean(stacked, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding

