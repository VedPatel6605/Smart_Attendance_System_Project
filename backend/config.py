import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'attendance_system')
    
    # Application Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    
    # Recognition Configuration
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.6))
    
    # Eye Detection Configuration
    EYE_ASPECT_RATIO_THRESHOLD = 0.25  # For blink detection
    EYE_AR_CONSEC_FRAMES = 3  # Frames to detect blink
    
    # Image Processing
    EYE_REGION_SIZE = (112, 112)  # Size for eye region crop
    MAX_SAMPLES_PER_USER = 10  # Maximum enrollment samples

