import mysql.connector
from mysql.connector import Error
import json
import numpy as np
from config import Config
from datetime import datetime

class Database:
    def __init__(self):
        self.config = Config()
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME
            )
            if self.connection.is_connected():
                print("Connected to MySQL database")
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise
    
    def get_connection(self):
        """Get database connection, reconnect if needed"""
        if self.connection is None or not self.connection.is_connected():
            self.connect()
        return self.connection
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.get_connection().cursor()
            
            # Create Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Users (
                    user_id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100),
                    enrollment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            # Create Embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Embeddings (
                    embedding_id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    embedding_vector BLOB NOT NULL,
                    sample_image_path VARCHAR(255),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
                )
            """)
            
            # Create Attendance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Attendance (
                    attendance_id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    device_id VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'Present',
                    confidence FLOAT,
                    liveness_status VARCHAR(20),
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_daily_attendance (user_id, date)
                )
            """)
            
            self.get_connection().commit()
            cursor.close()
            print("Database tables created successfully")
        except Error as e:
            print(f"Error creating tables: {e}")
            raise
    
    def insert_user(self, name, email=None, notes=None):
        """Insert a new user"""
        try:
            cursor = self.get_connection().cursor()
            cursor.execute("""
                INSERT INTO Users (name, email, notes)
                VALUES (%s, %s, %s)
            """, (name, email, notes))
            user_id = cursor.lastrowid
            self.get_connection().commit()
            cursor.close()
            return user_id
        except Error as e:
            print(f"Error inserting user: {e}")
            raise
    
    def insert_embedding(self, user_id, embedding_vector, sample_image_path=None):
        """Insert embedding for a user"""
        try:
            cursor = self.get_connection().cursor()
            # Convert numpy array to bytes
            embedding_bytes = embedding_vector.tobytes()
            cursor.execute("""
                INSERT INTO Embeddings (user_id, embedding_vector, sample_image_path)
                VALUES (%s, %s, %s)
            """, (user_id, embedding_bytes, sample_image_path))
            embedding_id = cursor.lastrowid
            self.get_connection().commit()
            cursor.close()
            return embedding_id
        except Error as e:
            print(f"Error inserting embedding: {e}")
            raise
    
    def get_user_embeddings(self, user_id):
        """Get all embeddings for a user"""
        try:
            cursor = self.get_connection().cursor()
            cursor.execute("""
                SELECT embedding_vector FROM Embeddings
                WHERE user_id = %s
            """, (user_id,))
            results = cursor.fetchall()
            embeddings = []
            for (embedding_bytes,) in results:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                embeddings.append(embedding)
            cursor.close()
            return embeddings
        except Error as e:
            print(f"Error getting embeddings: {e}")
            raise
    
    def get_all_users_with_embeddings(self):
        """Get all users who have embeddings"""
        try:
            cursor = self.get_connection().cursor(dictionary=True)
            cursor.execute("""
                SELECT DISTINCT u.user_id, u.name, u.email, u.enrollment_date
                FROM Users u
                INNER JOIN Embeddings e ON u.user_id = e.user_id
            """)
            users = cursor.fetchall()
            cursor.close()
            return users
        except Error as e:
            print(f"Error getting users: {e}")
            raise
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            cursor = self.get_connection().cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM Users WHERE user_id = %s
            """, (user_id,))
            user = cursor.fetchone()
            cursor.close()
            return user
        except Error as e:
            print(f"Error getting user: {e}")
            raise
    
    def mark_attendance(self, user_id, date, time, device_id=None, status='Present', 
                       confidence=None, liveness_status=None, notes=None):
        """Mark attendance for a user"""
        try:
            cursor = self.get_connection().cursor()
            cursor.execute("""
                INSERT INTO Attendance (user_id, date, time, device_id, status, 
                                      confidence, liveness_status, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    time = VALUES(time),
                    device_id = VALUES(device_id),
                    status = VALUES(status),
                    confidence = VALUES(confidence),
                    liveness_status = VALUES(liveness_status),
                    notes = VALUES(notes)
            """, (user_id, date, time, device_id, status, confidence, liveness_status, notes))
            attendance_id = cursor.lastrowid
            self.get_connection().commit()
            cursor.close()
            return attendance_id
        except Error as e:
            print(f"Error marking attendance: {e}")
            raise
    
    def get_attendance_by_date(self, date):
        """Get attendance records for a specific date"""
        try:
            cursor = self.get_connection().cursor(dictionary=True)
            cursor.execute("""
                SELECT a.*, u.name, u.email
                FROM Attendance a
                JOIN Users u ON a.user_id = u.user_id
                WHERE a.date = %s
                ORDER BY a.time
            """, (date,))
            records = cursor.fetchall()
            cursor.close()
            return records
        except Error as e:
            print(f"Error getting attendance: {e}")
            raise
    
    def get_user_attendance(self, user_id, start_date=None, end_date=None):
        """Get attendance records for a user"""
        try:
            cursor = self.get_connection().cursor(dictionary=True)
            if start_date and end_date:
                cursor.execute("""
                    SELECT * FROM Attendance
                    WHERE user_id = %s AND date BETWEEN %s AND %s
                    ORDER BY date DESC, time DESC
                """, (user_id, start_date, end_date))
            else:
                cursor.execute("""
                    SELECT * FROM Attendance
                    WHERE user_id = %s
                    ORDER BY date DESC, time DESC
                """, (user_id,))
            records = cursor.fetchall()
            cursor.close()
            return records
        except Error as e:
            print(f"Error getting user attendance: {e}")
            raise
    
    def delete_user(self, user_id):
        """Delete a user and all associated data"""
        try:
            cursor = self.get_connection().cursor()
            cursor.execute("DELETE FROM Users WHERE user_id = %s", (user_id,))
            self.get_connection().commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error deleting user: {e}")
            raise
    
    def get_all_users(self):
        """Get all users"""
        try:
            cursor = self.get_connection().cursor(dictionary=True)
            cursor.execute("SELECT * FROM Users ORDER BY enrollment_date DESC")
            users = cursor.fetchall()
            cursor.close()
            return users
        except Error as e:
            print(f"Error getting all users: {e}")
            raise

