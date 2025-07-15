"""
Database Logger for TITANUS CME Detection System
Handles logging of predictions, instrument data, and alerts to PostgreSQL/SQLite
"""

import json
from datetime import datetime
import psycopg2
from models import insert_prediction, insert_instrument_data, insert_alert_log

def log_prediction(conn, prediction_results, fused_data):
    """Log CME prediction results to database"""
    try:
        # Prepare prediction data for database
        prediction_data = {
            'cme_detected': prediction_results.get('cme_detected', False),
            'confidence': prediction_results.get('confidence', 0.0),
            'prediction_method': prediction_results.get('prediction_method', 'C_THRESHOLD_ENGINE'),
            'data_sources': fused_data.get('sources', []) if fused_data else [],
            'features': prediction_results.get('features', {}),
            'future_predictions': prediction_results.get('future_predictions', {}),
            'thresholds_used': prediction_results.get('thresholds_used', {})
        }
        
        # Insert prediction record
        prediction_id = insert_prediction(conn, prediction_data)
        
        # Log instrument data if available
        if fused_data and fused_data.get('features'):
            log_instrument_data(conn, fused_data)
        
        print(f"Prediction logged to database with ID: {prediction_id}")
        return prediction_id
        
    except Exception as e:
        print(f"Failed to log prediction to database: {e}")
        raise

def log_instrument_data(conn, fused_data):
    """Log instrument data to database"""
    try:
        features = fused_data.get('features', {})
        sources = fused_data.get('sources', [])
        
        # Create instrument data record
        instrument_data = {
            'data_source': ','.join(sources),
            'solar_wind_speed': features.get('solar_wind_speed'),
            'proton_density': features.get('proton_density'),
            'temperature': features.get('temperature'),
            'ion_flux': features.get('ion_flux'),
            'electron_flux': features.get('electron_flux'),
            'magnetic_field_x': features.get('magnetic_field_x'),
            'magnetic_field_y': features.get('magnetic_field_y'),
            'magnetic_field_z': features.get('magnetic_field_z'),
            'magnetic_field_magnitude': features.get('magnetic_field_magnitude'),
            'dynamic_pressure': features.get('dynamic_pressure')
        }
        
        # Insert for each instrument type
        for instrument_type in sources:
            data_id = insert_instrument_data(conn, instrument_type, instrument_data)
            print(f"Instrument data logged for {instrument_type} with ID: {data_id}")
        
    except Exception as e:
        print(f"Failed to log instrument data: {e}")
        raise

def log_alert(conn, alert_type, message, recipients, status='sent'):
    """Log alert to database"""
    try:
        alert_id = insert_alert_log(conn, alert_type, message, recipients, status)
        print(f"Alert logged with ID: {alert_id}")
        return alert_id
        
    except Exception as e:
        print(f"Failed to log alert: {e}")
        raise

def log_threshold_update(conn, threshold_name, old_value, new_value, accuracy, notes=""):
    """Log threshold updates to database"""
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO threshold_history 
            (threshold_name, threshold_value, training_accuracy, notes)
            VALUES (%s, %s, %s, %s)
        """, (threshold_name, new_value, accuracy, 
              f"Updated from {old_value} to {new_value}. {notes}"))
        
        conn.commit()
        threshold_id = cursor.lastrowid if hasattr(cursor, 'lastrowid') else cursor.rowcount
        cursor.close()
        
        print(f"Threshold update logged with ID: {threshold_id}")
        return threshold_id
        
    except Exception as e:
        print(f"Failed to log threshold update: {e}")
        conn.rollback()
        raise

def get_recent_predictions(conn, limit=10):
    """Get recent predictions from database"""
    try:
        from models import get_recent_predictions
        return get_recent_predictions(conn, limit)
        
    except Exception as e:
        print(f"Failed to get recent predictions: {e}")
        return []

def get_prediction_statistics(conn, days=30):
    """Get prediction statistics for the last N days"""
    try:
        cursor = conn.cursor()
        
        # Check if using PostgreSQL or SQLite
        is_postgres = hasattr(conn, 'autocommit')
        
        if is_postgres:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN cme_detected THEN 1 ELSE 0 END) as cme_detections,
                    AVG(confidence) as avg_confidence,
                    MAX(confidence) as max_confidence,
                    COUNT(DISTINCT prediction_method) as methods_used
                FROM cme_predictions 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
            """, (days,))
        else:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN cme_detected THEN 1 ELSE 0 END) as cme_detections,
                    AVG(confidence) as avg_confidence,
                    MAX(confidence) as max_confidence,
                    COUNT(DISTINCT prediction_method) as methods_used
                FROM cme_predictions 
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
            """, (days,))
        
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            return {
                'total_predictions': result[0] or 0,
                'cme_detections': result[1] or 0,
                'avg_confidence': result[2] or 0.0,
                'max_confidence': result[3] or 0.0,
                'methods_used': result[4] or 0,
                'detection_rate': (result[1] or 0) / max(1, result[0] or 1) * 100
            }
        else:
            return {
                'total_predictions': 0,
                'cme_detections': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'methods_used': 0,
                'detection_rate': 0.0
            }
        
    except Exception as e:
        print(f"Failed to get prediction statistics: {e}")
        return {}

def export_data_to_csv(conn, table_name, output_file, days=30):
    """Export database data to CSV file"""
    try:
        import pandas as pd
        
        cursor = conn.cursor()
        
        # Check if using PostgreSQL or SQLite
        is_postgres = hasattr(conn, 'autocommit')
        
        if table_name == 'cme_predictions':
            if is_postgres:
                query = """
                    SELECT * FROM cme_predictions 
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY timestamp DESC
                """
            else:
                query = """
                    SELECT * FROM cme_predictions 
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp DESC
                """
        elif table_name == 'instruments_data':
            if is_postgres:
                query = """
                    SELECT * FROM instruments_data 
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                    ORDER BY timestamp DESC
                """
            else:
                query = """
                    SELECT * FROM instruments_data 
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp DESC
                """
        else:
            raise ValueError(f"Unknown table: {table_name}")
        
        cursor.execute(query, (days,))
        
        # Get column names
        if is_postgres:
            columns = [desc[0] for desc in cursor.description]
        else:
            columns = [desc[0] for desc in cursor.description]
        
        # Fetch all data
        data = cursor.fetchall()
        cursor.close()
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        
        print(f"Exported {len(df)} records to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Failed to export data to CSV: {e}")
        return None

def cleanup_old_data(conn, days_to_keep=365):
    """Clean up old data from database"""
    try:
        cursor = conn.cursor()
        
        # Check if using PostgreSQL or SQLite
        is_postgres = hasattr(conn, 'autocommit')
        
        tables_to_clean = ['cme_predictions', 'instruments_data', 'alert_logs']
        
        total_deleted = 0
        
        for table in tables_to_clean:
            if is_postgres:
                cursor.execute(f"""
                    DELETE FROM {table} 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """, (days_to_keep,))
            else:
                cursor.execute(f"""
                    DELETE FROM {table} 
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (days_to_keep,))
            
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            print(f"Deleted {deleted_count} old records from {table}")
        
        conn.commit()
        cursor.close()
        
        print(f"Total deleted records: {total_deleted}")
        return total_deleted
        
    except Exception as e:
        print(f"Failed to cleanup old data: {e}")
        conn.rollback()
        return 0

def log_system_event(conn, event_type, description, severity='INFO'):
    """Log system events"""
    try:
        # Use alert_logs table for system events
        log_alert(conn, f"SYSTEM_{event_type}", description, "system", "logged")
        
    except Exception as e:
        print(f"Failed to log system event: {e}")

def get_data_quality_report(conn):
    """Generate data quality report"""
    try:
        cursor = conn.cursor()
        
        # Check if using PostgreSQL or SQLite
        is_postgres = hasattr(conn, 'autocommit')
        
        report = {}
        
        # Instrument data quality
        if is_postgres:
            cursor.execute("""
                SELECT 
                    instrument_type,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN solar_wind_speed IS NULL THEN 1 END) as missing_speed,
                    COUNT(CASE WHEN proton_density IS NULL THEN 1 END) as missing_density,
                    COUNT(CASE WHEN magnetic_field_magnitude IS NULL THEN 1 END) as missing_mag
                FROM instruments_data 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY instrument_type
            """)
        else:
            cursor.execute("""
                SELECT 
                    instrument_type,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN solar_wind_speed IS NULL THEN 1 END) as missing_speed,
                    COUNT(CASE WHEN proton_density IS NULL THEN 1 END) as missing_density,
                    COUNT(CASE WHEN magnetic_field_magnitude IS NULL THEN 1 END) as missing_mag
                FROM instruments_data 
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY instrument_type
            """)
        
        instrument_quality = cursor.fetchall()
        report['instrument_data_quality'] = []
        
        for row in instrument_quality:
            instrument_type, total, missing_speed, missing_density, missing_mag = row
            quality_score = 100 * (1 - (missing_speed + missing_density + missing_mag) / (3 * total))
            
            report['instrument_data_quality'].append({
                'instrument': instrument_type,
                'total_records': total,
                'quality_score': quality_score,
                'missing_data': {
                    'speed': missing_speed,
                    'density': missing_density,
                    'magnetic_field': missing_mag
                }
            })
        
        # Prediction quality
        if is_postgres:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN confidence < 0.3 THEN 1 END) as low_confidence,
                    COUNT(CASE WHEN confidence > 0.8 THEN 1 END) as high_confidence
                FROM cme_predictions 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            """)
        else:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN confidence < 0.3 THEN 1 END) as low_confidence,
                    COUNT(CASE WHEN confidence > 0.8 THEN 1 END) as high_confidence
                FROM cme_predictions 
                WHERE timestamp >= datetime('now', '-7 days')
            """)
        
        pred_result = cursor.fetchone()
        if pred_result:
            total_pred, avg_conf, low_conf, high_conf = pred_result
            report['prediction_quality'] = {
                'total_predictions': total_pred or 0,
                'average_confidence': avg_conf or 0.0,
                'low_confidence_count': low_conf or 0,
                'high_confidence_count': high_conf or 0,
                'confidence_distribution': {
                    'low': (low_conf or 0) / max(1, total_pred or 1) * 100,
                    'high': (high_conf or 0) / max(1, total_pred or 1) * 100
                }
            }
        
        cursor.close()
        return report
        
    except Exception as e:
        print(f"Failed to generate data quality report: {e}")
        return {}

if __name__ == "__main__":
    # Test database logging functionality
    from models import init_database
    
    try:
        conn = init_database()
        
        # Test logging a sample prediction
        sample_prediction = {
            'cme_detected': True,
            'confidence': 0.85,
            'prediction_method': 'TEST',
            'features': {'solar_wind_speed': 650, 'proton_density': 12},
            'future_predictions': {'speed': [640, 630, 620]},
            'thresholds_used': {'SOLAR_WIND_THRESHOLD': 600}
        }
        
        sample_fused_data = {
            'sources': ['swis', 'magnetometer'],
            'features': {
                'solar_wind_speed': 650,
                'proton_density': 12,
                'magnetic_field_magnitude': 18
            }
        }
        
        prediction_id = log_prediction(conn, sample_prediction, sample_fused_data)
        print(f"Test prediction logged with ID: {prediction_id}")
        
        # Get statistics
        stats = get_prediction_statistics(conn, 30)
        print(f"Prediction statistics: {stats}")
        
        # Generate quality report
        quality_report = get_data_quality_report(conn)
        print(f"Data quality report: {quality_report}")
        
        conn.close()
        print("Database logging test completed successfully!")
        
    except Exception as e:
        print(f"Database logging test failed: {e}")
