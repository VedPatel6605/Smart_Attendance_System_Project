#!/usr/bin/env python3
"""
Database initialization script
Run this script to create the database and tables in MySQL
"""

import mysql.connector
from mysql.connector import Error
from config import Config
import sys

def create_database():
    """Create database if it doesn't exist"""
    config = Config()
    
    try:
        # Connect without specifying database
        connection = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME}")
            print(f"Database '{config.DB_NAME}' created or already exists")
            cursor.close()
            connection.close()
            
    except Error as e:
        print(f"Error creating database: {e}")
        sys.exit(1)

def main():
    print("Initializing database...")
    create_database()
    
    # Import database module to create tables
    from database import Database
    db = Database()
    db.create_tables()
    print("Database initialization complete!")

if __name__ == "__main__":
    main()

