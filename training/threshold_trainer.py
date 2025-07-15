"""
Threshold Training Module for TITANUS CME Detection
Uses CACTUS CME catalog data to optimize detection thresholds
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

class ThresholdTrainer:
    def __init__(self, cactus_file='data/CACTUS_events.csv'):
        self.cactus_file = cactus_file
        self.current_thresholds = self.load_current_thresholds()
        self.training_data = None
        self.scaler = StandardScaler()
        
    def load_current_thresholds(self):
        """Load current thresholds from C header file"""
        thresholds = {
            'SOLAR_WIND_THRESHOLD': 600.0,
            'PROTON_DENSITY_THRESHOLD': 10.0,
            'TEMPERATURE_THRESHOLD': 50000.0,
            'DYNAMIC_PRESSURE_THRESHOLD': 5.0,
            'ION_FLUX_THRESHOLD': 1e6,
            'ELECTRON_FLUX_THRESHOLD': 1e7,
            'MAGNETIC_ROTATION_THRESHOLD': 30.0,
            'MAGNETIC_FIELD_THRESHOLD': 20.0,
            'MIN_DETECTION_FLAGS': 3
        }
        
        # Try to read from header file
        try:
            with open('c_core/model_weights.h', 'r') as f:
                content = f.read()
                for key in thresholds.keys():
                    if f'#define {key}' in content:
                        line = [l for l in content.split('\n') if f'#define {key}' in l][0]
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                value = float(parts[2])
                                thresholds[key] = value
                            except ValueError:
                                pass
        except FileNotFoundError:
            print("Warning: model_weights.h not found, using default thresholds")
        
        return thresholds
    
    def load_cactus_data(self):
        """Load CACTUS CME catalog data"""
        try:
            df = pd.read_csv(self.cactus_file)
            print(f"Loaded {len(df)} CACTUS events")
            return df
        except FileNotFoundError:
            print("CACTUS data file not found, generating synthetic training data")
            return self.generate_synthetic_training_data()
    
    def generate_synthetic_training_data(self):
        """Generate synthetic training data for threshold optimization"""
        print("Generating synthetic training data...")
        
        n_samples = 1000
        n_cme = 200  # 20% CME events
        
        data = []
        
        for i in range(n_samples):
            is_cme = i < n_cme
            
            if is_cme:
                # CME event characteristics
                sample = {
                    'cme_detected': 1,
                    'solar_wind_speed': np.random.normal(650, 150),
                    'proton_density': np.random.lognormal(2.5, 0.8),
                    'temperature': np.random.normal(80000, 30000),
                    'dynamic_pressure': np.random.lognormal(2.0, 0.8),
                    'ion_flux': np.random.lognormal(14, 1.5),
                    'electron_flux': np.random.lognormal(16, 1.5),
                    'magnetic_field_magnitude': np.random.normal(18, 8),
                    'magnetic_rotation': np.random.uniform(20, 180),
                    'flux_dropout': np.random.uniform(0.3, 0.9),
                    'temperature_depression': np.random.uniform(0.3, 0.8)
                }
            else:
                # Quiet solar wind characteristics
                sample = {
                    'cme_detected': 0,
                    'solar_wind_speed': np.random.normal(420, 80),
                    'proton_density': np.random.lognormal(1.6, 0.6),
                    'temperature': np.random.normal(120000, 40000),
                    'dynamic_pressure': np.random.lognormal(1.2, 0.6),
                    'ion_flux': np.random.lognormal(12, 1.0),
                    'electron_flux': np.random.lognormal(14, 1.0),
                    'magnetic_field_magnitude': np.random.normal(6, 3),
                    'magnetic_rotation': np.random.uniform(0, 30),
                    'flux_dropout': np.random.uniform(0, 0.4),
                    'temperature_depression': np.random.uniform(0, 0.3)
                }
            
            # Apply physical constraints
            sample['solar_wind_speed'] = max(200, min(2000, sample['solar_wind_speed']))
            sample['proton_density'] = max(0.1, min(100, sample['proton_density']))
            sample['temperature'] = max(10000, min(1e6, sample['temperature']))
            sample['dynamic_pressure'] = max(0.1, min(50, sample['dynamic_pressure']))
            sample['ion_flux'] = max(1e3, min(1e9, sample['ion_flux']))
            sample['electron_flux'] = max(1e4, min(1e10, sample['electron_flux']))
            sample['magnetic_field_magnitude'] = max(0.1, min(100, sample['magnetic_field_magnitude']))
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def prepare_training_data(self, cactus_df):
        """Prepare training data from CACTUS events"""
        # Extract features for training
        features = []
        labels = []
        
        for _, row in cactus_df.iterrows():
            # Create feature vector
            feature_vector = [
                row.get('solar_wind_speed', 400),
                row.get('proton_density', 5),
                row.get('temperature', 100000),
                row.get('dynamic_pressure', 2),
                row.get('ion_flux', 1e5),
                row.get('electron_flux', 1e6),
                row.get('magnetic_field_magnitude', 6),
                row.get('magnetic_rotation', 10),
                row.get('flux_dropout', 0.1),
                row.get('temperature_depression', 0.1)
            ]
            
            features.append(feature_vector)
            labels.append(row.get('cme_detected', 0))
        
        return np.array(features), np.array(labels)
    
    def optimize_thresholds_grid_search(self, features, labels):
        """Optimize thresholds using grid search"""
        print("Optimizing thresholds using grid search...")
        
        # Define threshold ranges for optimization
        threshold_ranges = {
            'SOLAR_WIND_THRESHOLD': np.linspace(450, 800, 20),
            'PROTON_DENSITY_THRESHOLD': np.linspace(5, 20, 16),
            'TEMPERATURE_THRESHOLD': np.linspace(30000, 80000, 20),
            'DYNAMIC_PRESSURE_THRESHOLD': np.linspace(2, 10, 17),
            'ION_FLUX_THRESHOLD': np.logspace(5, 7, 15),
            'ELECTRON_FLUX_THRESHOLD': np.logspace(6, 8, 15),
            'MAGNETIC_FIELD_THRESHOLD': np.linspace(8, 25, 18),
            'MAGNETIC_ROTATION_THRESHOLD': np.linspace(15, 45, 16)
        }
        
        best_score = 0
        best_thresholds = self.current_thresholds.copy()
        
        # Grid search for each threshold independently
        for threshold_name, threshold_values in threshold_ranges.items():
            print(f"Optimizing {threshold_name}...")
            
            threshold_scores = []
            for threshold_value in threshold_values:
                # Test this threshold value
                test_thresholds = self.current_thresholds.copy()
                test_thresholds[threshold_name] = threshold_value
                
                # Apply thresholds to get predictions
                predictions = self.apply_thresholds(features, test_thresholds)
                
                # Calculate F1 score
                f1 = f1_score(labels, predictions, zero_division=0)
                threshold_scores.append(f1)
            
            # Find best threshold for this parameter
            best_idx = np.argmax(threshold_scores)
            best_value = threshold_values[best_idx]
            best_f1 = threshold_scores[best_idx]
            
            if best_f1 > best_score:
                best_thresholds[threshold_name] = best_value
                best_score = best_f1
                print(f"  New best {threshold_name}: {best_value:.3f} (F1: {best_f1:.3f})")
        
        return best_thresholds, best_score
    
    def apply_thresholds(self, features, thresholds):
        """Apply threshold-based detection to features"""
        predictions = []
        
        for feature_vector in features:
            detection_flags = 0
            
            # Check each threshold
            if feature_vector[0] > thresholds['SOLAR_WIND_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[1] > thresholds['PROTON_DENSITY_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[2] < thresholds['TEMPERATURE_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[3] > thresholds['DYNAMIC_PRESSURE_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[4] > thresholds['ION_FLUX_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[5] > thresholds['ELECTRON_FLUX_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[6] > thresholds['MAGNETIC_FIELD_THRESHOLD']:
                detection_flags += 1
            
            if feature_vector[7] > thresholds['MAGNETIC_ROTATION_THRESHOLD']:
                detection_flags += 1
            
            # CME detected if enough flags triggered
            cme_detected = detection_flags >= thresholds['MIN_DETECTION_FLAGS']
            predictions.append(1 if cme_detected else 0)
        
        return np.array(predictions)
    
    def train_ml_model(self, features, labels):
        """Train machine learning model for feature importance"""
        print("Training ML model for feature importance analysis...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"ML Model Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Feature importance
        feature_names = [
            'solar_wind_speed', 'proton_density', 'temperature',
            'dynamic_pressure', 'ion_flux', 'electron_flux',
            'magnetic_field_magnitude', 'magnetic_rotation',
            'flux_dropout', 'temperature_depression'
        ]
        
        importance_scores = rf.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores))
        
        print("\nFeature Importance:")
        for feature, importance in sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.3f}")
        
        return rf, feature_importance, {'accuracy': accuracy, 'f1': f1}
    
    def evaluate_thresholds(self, features, labels, thresholds):
        """Evaluate threshold performance"""
        predictions = self.apply_thresholds(features, thresholds)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def train_thresholds(self):
        """Main training function"""
        print("Starting threshold training...")
        
        # Load training data
        cactus_df = self.load_cactus_data()
        features, labels = self.prepare_training_data(cactus_df)
        
        print(f"Training data: {len(features)} samples, {sum(labels)} CME events")
        
        # Evaluate current thresholds
        current_performance = self.evaluate_thresholds(features, labels, self.current_thresholds)
        print(f"Current threshold performance: F1={current_performance['f1_score']:.3f}")
        
        # Optimize thresholds
        optimized_thresholds, best_score = self.optimize_thresholds_grid_search(features, labels)
        
        # Evaluate optimized thresholds
        optimized_performance = self.evaluate_thresholds(features, labels, optimized_thresholds)
        print(f"Optimized threshold performance: F1={optimized_performance['f1_score']:.3f}")
        
        # Train ML model for comparison
        ml_model, feature_importance, ml_performance = self.train_ml_model(features, labels)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'current_thresholds': self.current_thresholds,
            'optimized_thresholds': optimized_thresholds,
            'current_performance': current_performance,
            'optimized_performance': optimized_performance,
            'ml_performance': ml_performance,
            'feature_importance': feature_importance,
            'training_samples': len(features),
            'cme_events': int(sum(labels))
        }
        
        # Save training results
        os.makedirs('logs', exist_ok=True)
        with open('logs/threshold_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return optimized_thresholds

def update_c_thresholds(new_thresholds):
    """Update C header file with new thresholds"""
    try:
        # Read current header file
        with open('c_core/model_weights.h', 'r') as f:
            content = f.read()
        
        # Update threshold values
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            updated = False
            for threshold_name, threshold_value in new_thresholds.items():
                if line.startswith(f'#define {threshold_name}'):
                    # Update the line
                    parts = line.split()
                    if len(parts) >= 3:
                        comment = ' '.join(parts[3:]) if len(parts) > 3 else ''
                        new_line = f"#define {threshold_name}      {threshold_value}     {comment}"
                        updated_lines.append(new_line)
                        updated = True
                        break
            
            if not updated:
                updated_lines.append(line)
        
        # Write updated content
        with open('c_core/model_weights.h', 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print("C header file updated with new thresholds")
        
        # Log the update
        from models import get_db_connection
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for threshold_name, threshold_value in new_thresholds.items():
                cursor.execute("""
                    INSERT INTO threshold_history (threshold_name, threshold_value, notes)
                    VALUES (%s, %s, %s)
                """, (threshold_name, float(threshold_value), 'Automated threshold update'))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Failed to log threshold update to database: {e}")
        
    except Exception as e:
        print(f"Failed to update C header file: {e}")
        raise

def train_thresholds():
    """Main training function for external use"""
    trainer = ThresholdTrainer()
    return trainer.train_thresholds()

if __name__ == "__main__":
    # Run threshold training
    trainer = ThresholdTrainer()
    new_thresholds = trainer.train_thresholds()
    
    print("\nOptimized Thresholds:")
    for name, value in new_thresholds.items():
        print(f"{name}: {value}")
    
    # Update C header file
    update_c_thresholds(new_thresholds)
    print("\nThreshold training completed successfully!")
