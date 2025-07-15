"""
Database models for TITANUS CME Detection System
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        # Try PostgreSQL first
        conn = psycopg2.connect(
            host=os.getenv('PGHOST', 'localhost'),
            database=os.getenv('PGDATABASE', 'titanus'),
            user=os.getenv('PGUSER', 'postgres'),
            password=os.getenv('PGPASSWORD', 'password'),
            port=os.getenv('PGPORT', '5432')
        )
        return conn
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        # Fallback to SQLite
        import sqlite3
        conn = sqlite3.connect('logs/logs.db')
        conn.row_factory = sqlite3.Row
        return conn

def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    
    # Check if we're using PostgreSQL or SQLite
    is_postgres = hasattr(conn, 'autocommit')
    
    cursor = conn.cursor()
    
    try:
        if is_postgres:
            # PostgreSQL table creation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instruments_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    instrument_type VARCHAR(50) NOT NULL,
                    data_source VARCHAR(100),
                    solar_wind_speed REAL,
                    proton_density REAL,
                    temperature REAL,
                    ion_flux REAL,
                    electron_flux REAL,
                    magnetic_field_x REAL,
                    magnetic_field_y REAL,
                    magnetic_field_z REAL,
                    magnetic_field_magnitude REAL,
                    dynamic_pressure REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cme_predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cme_detected BOOLEAN DEFAULT FALSE,
                    confidence REAL,
                    prediction_method VARCHAR(100),
                    data_sources TEXT,
                    features_json TEXT,
                    future_predictions_json TEXT,
                    thresholds_used_json TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    alert_type VARCHAR(50),
                    message TEXT,
                    recipients TEXT,
                    status VARCHAR(20) DEFAULT 'sent'
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    threshold_name VARCHAR(100),
                    threshold_value REAL,
                    training_accuracy REAL,
                    notes TEXT
                )
            """)
            
        else:
            # SQLite table creation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instruments_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    instrument_type TEXT NOT NULL,
                    data_source TEXT,
                    solar_wind_speed REAL,
                    proton_density REAL,
                    temperature REAL,
                    ion_flux REAL,
                    electron_flux REAL,
                    magnetic_field_x REAL,
                    magnetic_field_y REAL,
                    magnetic_field_z REAL,
                    magnetic_field_magnitude REAL,
                    dynamic_pressure REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cme_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cme_detected BOOLEAN DEFAULT 0,
                    confidence REAL,
                    prediction_method TEXT,
                    data_sources TEXT,
                    features_json TEXT,
                    future_predictions_json TEXT,
                    thresholds_used_json TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    message TEXT,
                    recipients TEXT,
                    status TEXT DEFAULT 'sent'
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threshold_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    threshold_name TEXT,
                    threshold_value REAL,
                    training_accuracy REAL,
                    notes TEXT
                )
            """)
        
        conn.commit()
        print("Database tables initialized successfully")
        return conn
        
    except Exception as e:
        print(f"Database initialization error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def insert_instrument_data(conn, instrument_type, data_dict):
    """Insert instrument data into database"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO instruments_data 
            (instrument_type, data_source, solar_wind_speed, proton_density, 
             temperature, ion_flux, electron_flux, magnetic_field_x, 
             magnetic_field_y, magnetic_field_z, magnetic_field_magnitude, 
             dynamic_pressure)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            instrument_type,
            data_dict.get('data_source'),
            data_dict.get('solar_wind_speed'),
            data_dict.get('proton_density'),
            data_dict.get('temperature'),
            data_dict.get('ion_flux'),
            data_dict.get('electron_flux'),
            data_dict.get('magnetic_field_x'),
            data_dict.get('magnetic_field_y'),
            data_dict.get('magnetic_field_z'),
            data_dict.get('magnetic_field_magnitude'),
            data_dict.get('dynamic_pressure')
        ))
        
        conn.commit()
        return cursor.lastrowid if hasattr(cursor, 'lastrowid') else cursor.rowcount
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to insert instrument data: {e}")
    finally:
        cursor.close()

def insert_prediction(conn, prediction_data):
    """Insert CME prediction into database"""
    cursor = conn.cursor()
    
    try:
        import json
        
        cursor.execute("""
            INSERT INTO cme_predictions 
            (cme_detected, confidence, prediction_method, data_sources, 
             features_json, future_predictions_json, thresholds_used_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            prediction_data.get('cme_detected', False),
            prediction_data.get('confidence', 0.0),
            prediction_data.get('prediction_method', 'C_ENGINE'),
            json.dumps(prediction_data.get('data_sources', [])),
            json.dumps(prediction_data.get('features', {})),
            json.dumps(prediction_data.get('future_predictions', {})),
            json.dumps(prediction_data.get('thresholds_used', {}))
        ))
        
        conn.commit()
        return cursor.lastrowid if hasattr(cursor, 'lastrowid') else cursor.rowcount
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to insert prediction: {e}")
    finally:
        cursor.close()

def insert_alert_log(conn, alert_type, message, recipients, status='sent'):
    """Insert alert log into database"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO alert_logs (alert_type, message, recipients, status)
            VALUES (%s, %s, %s, %s)
        """, (alert_type, message, recipients, status))
        
        conn.commit()
        return cursor.lastrowid if hasattr(cursor, 'lastrowid') else cursor.rowcount
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to insert alert log: {e}")
    finally:
        cursor.close()

def get_recent_predictions(conn, limit=10):
    """Get recent CME predictions"""
    cursor = conn.cursor(cursor_factory=RealDictCursor if hasattr(conn, 'autocommit') else None)
    
    try:
        cursor.execute("""
            SELECT * FROM cme_predictions 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (limit,))
        
        return cursor.fetchall()
        
    except Exception as e:
        raise Exception(f"Failed to get recent predictions: {e}")
    finally:
        cursor.close()

def get_instrument_data_range(conn, instrument_type, start_time, end_time):
    """Get instrument data within time range"""
    cursor = conn.cursor(cursor_factory=RealDictCursor if hasattr(conn, 'autocommit') else None)
    
    try:
        cursor.execute("""
            SELECT * FROM instruments_data 
            WHERE instrument_type = %s 
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
        """, (instrument_type, start_time, end_time))
        
        return cursor.fetchall()
        
    except Exception as e:
        raise Exception(f"Failed to get instrument data: {e}")
    finally:
        cursor.close()
