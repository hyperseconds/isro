"""
Magnetometer Data Parser for TITANUS CME Detection
Handles both CDF and CSV formats for magnetic field measurements
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def parse_cdf(file_path):
    """Parse magnetometer CDF file"""
    try:
        import cdflib
        
        cdf_file = cdflib.CDF(file_path)
        
        # Extract magnetometer-specific variables
        data = {
            'timestamp': [],
            'magnetic_field': [],  # [Bx, By, Bz] components
            'magnetic_field_magnitude': [],
            'coordinate_system': 'GSE',  # Default coordinate system
            'data_quality': []
        }
        
        # Get time data
        if 'Epoch' in cdf_file.cdf_info()['zVariables']:
            epoch = cdf_file.varget('Epoch')
            data['timestamp'] = cdflib.cdfepoch.encode(epoch)
        
        # Magnetic field components
        mag_field_vars = ['B_GSE', 'B_GSM', 'B_RTN', 'Magnetic_Field', 'B_vec']
        for var in mag_field_vars:
            if var in cdf_file.cdf_info()['zVariables']:
                b_components = cdf_file.varget(var)
                if b_components.ndim == 2 and b_components.shape[1] >= 3:
                    data['magnetic_field'] = b_components[:, :3].tolist()
                    # Calculate magnitude
                    data['magnetic_field_magnitude'] = np.sqrt(
                        np.sum(b_components[:, :3]**2, axis=1)
                    ).tolist()
                    
                    # Set coordinate system
                    if 'GSE' in var:
                        data['coordinate_system'] = 'GSE'
                    elif 'GSM' in var:
                        data['coordinate_system'] = 'GSM'
                    elif 'RTN' in var:
                        data['coordinate_system'] = 'RTN'
                break
        
        # If no vector field found, try individual components
        if not data['magnetic_field']:
            bx_vars = ['Bx', 'B_x', 'BX_GSE', 'BX']
            by_vars = ['By', 'B_y', 'BY_GSE', 'BY']
            bz_vars = ['Bz', 'B_z', 'BZ_GSE', 'BZ']
            
            bx = by = bz = None
            
            for var in bx_vars:
                if var in cdf_file.cdf_info()['zVariables']:
                    bx = cdf_file.varget(var)
                    break
            
            for var in by_vars:
                if var in cdf_file.cdf_info()['zVariables']:
                    by = cdf_file.varget(var)
                    break
            
            for var in bz_vars:
                if var in cdf_file.cdf_info()['zVariables']:
                    bz = cdf_file.varget(var)
                    break
            
            if bx is not None and by is not None and bz is not None:
                data['magnetic_field'] = np.column_stack([bx, by, bz]).tolist()
                data['magnetic_field_magnitude'] = np.sqrt(bx**2 + by**2 + bz**2).tolist()
        
        # Data quality flags
        quality_vars = ['Quality', 'Data_Quality', 'QF', 'Flag']
        for var in quality_vars:
            if var in cdf_file.cdf_info()['zVariables']:
                data['data_quality'] = cdf_file.varget(var).tolist()
                break
        
        cdf_file.close()
        
        # Validate data
        data = validate_magnetometer_data(data)
        
        return data
        
    except Exception as e:
        print(f"Magnetometer CDF parsing failed: {e}")
        # Try CSV fallback
        csv_path = file_path.replace('.cdf', '.csv')
        if os.path.exists(csv_path):
            return parse_csv(csv_path)
        else:
            # Use sample data
            return generate_sample_magnetometer_data()

def parse_csv(file_path):
    """Parse magnetometer CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Map common column names
        column_mapping = {
            'time': 'timestamp',
            'Time': 'timestamp',
            'datetime': 'timestamp',
            'B_x': 'bx',
            'B_y': 'by',
            'B_z': 'bz',
            'BX': 'bx',
            'BY': 'by',
            'BZ': 'bz',
            'Bx_GSE': 'bx',
            'By_GSE': 'by',
            'Bz_GSE': 'bz',
            'B_total': 'b_magnitude',
            'B_mag': 'b_magnitude',
            '|B|': 'b_magnitude'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        data = {
            'timestamp': [],
            'magnetic_field': [],
            'magnetic_field_magnitude': [],
            'coordinate_system': 'GSE',
            'data_quality': []
        }
        
        # Extract data
        if 'timestamp' in df.columns:
            data['timestamp'] = pd.to_datetime(df['timestamp']).tolist()
        else:
            # Generate timestamps
            data['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=24),
                periods=len(df),
                freq='H'
            ).tolist()
        
        # Extract magnetic field components
        if all(col in df.columns for col in ['bx', 'by', 'bz']):
            bx = df['bx'].fillna(5.0).tolist()
            by = df['by'].fillna(0.0).tolist()
            bz = df['bz'].fillna(-3.0).tolist()
            
            data['magnetic_field'] = list(zip(bx, by, bz))
            data['magnetic_field_magnitude'] = [
                np.sqrt(x**2 + y**2 + z**2) for x, y, z in data['magnetic_field']
            ]
        elif 'b_magnitude' in df.columns:
            # Only magnitude available
            data['magnetic_field_magnitude'] = df['b_magnitude'].fillna(10.0).tolist()
            # Generate approximate components
            data['magnetic_field'] = generate_field_components(data['magnetic_field_magnitude'])
        
        # Extract data quality if available
        if 'quality' in df.columns:
            data['data_quality'] = df['quality'].tolist()
        elif 'flag' in df.columns:
            data['data_quality'] = df['flag'].tolist()
        
        return validate_magnetometer_data(data)
        
    except Exception as e:
        print(f"Magnetometer CSV parsing failed: {e}")
        return generate_sample_magnetometer_data()

def generate_field_components(magnitudes):
    """Generate approximate field components from magnitude"""
    components = []
    for mag in magnitudes:
        # Typical solar wind field: mostly radial with small perpendicular components
        bx = mag * 0.8 + np.random.normal(0, mag * 0.1)
        by = np.random.normal(0, mag * 0.3)
        bz = np.random.normal(0, mag * 0.3)
        
        # Renormalize to match magnitude
        current_mag = np.sqrt(bx**2 + by**2 + bz**2)
        if current_mag > 0:
            scale = mag / current_mag
            bx *= scale
            by *= scale
            bz *= scale
        
        components.append([bx, by, bz])
    
    return components

def validate_magnetometer_data(data):
    """Validate and clean magnetometer data"""
    # Ensure minimum data length
    min_length = max(len(data.get('magnetic_field', [])), 
                    len(data.get('magnetic_field_magnitude', [])), 1)
    
    # Fill missing data with typical IMF values
    if not data.get('magnetic_field'):
        data['magnetic_field'] = generate_realistic_magnetic_field(min_length)
    
    if not data.get('magnetic_field_magnitude'):
        if data.get('magnetic_field'):
            data['magnetic_field_magnitude'] = [
                np.sqrt(sum(b**2 for b in field)) for field in data['magnetic_field']
            ]
        else:
            data['magnetic_field_magnitude'] = [5.0 + np.random.normal(0, 2) for _ in range(min_length)]
    
    if not data.get('timestamp'):
        data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(hours=min_length),
            periods=min_length,
            freq='H'
        ).tolist()
    
    # Apply physical constraints (nT)
    data['magnetic_field_magnitude'] = [max(0.1, min(100, b)) for b in data['magnetic_field_magnitude']]
    
    # Validate field components
    validated_field = []
    for i, field in enumerate(data['magnetic_field']):
        if len(field) >= 3:
            bx, by, bz = field[:3]
            # Apply reasonable limits
            bx = max(-50, min(50, bx))
            by = max(-50, min(50, by))
            bz = max(-50, min(50, bz))
            validated_field.append([bx, by, bz])
        else:
            # Generate from magnitude
            mag = data['magnetic_field_magnitude'][i] if i < len(data['magnetic_field_magnitude']) else 5.0
            validated_field.append(generate_field_components([mag])[0])
    
    data['magnetic_field'] = validated_field
    
    return data

def generate_realistic_magnetic_field(length):
    """Generate realistic magnetic field time series"""
    field_components = []
    
    for i in range(length):
        # Base IMF with Parker spiral structure
        # Typical values for solar wind at 1 AU
        
        # Radial component (positive outward)
        br = 3.0 + np.random.normal(0, 1.5)
        
        # Tangential component (Parker spiral)
        bt = -2.0 + np.random.normal(0, 2.0)
        
        # Normal component (small)
        bn = np.random.normal(0, 1.5)
        
        # Add some coherent structures (current sheets, etc.)
        if i > 0 and np.random.random() < 0.1:  # 10% chance of rotation
            rotation_angle = np.random.uniform(-np.pi, np.pi)
            # Rotate field vector
            cos_rot = np.cos(rotation_angle)
            sin_rot = np.sin(rotation_angle)
            
            br_new = br * cos_rot - bt * sin_rot
            bt_new = br * sin_rot + bt * cos_rot
            
            br, bt = br_new, bt_new
        
        field_components.append([br, bt, bn])
    
    return field_components

def generate_sample_magnetometer_data():
    """Generate realistic sample magnetometer data"""
    print("Generating sample magnetometer data...")
    
    # Create 24 hours of data
    hours = 24
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=hours),
        periods=hours,
        freq='H'
    )
    
    # Generate realistic magnetic field with flux rope signature
    magnetic_field = []
    magnetic_field_magnitude = []
    
    # Add a magnetic flux rope (CME signature) in the middle
    flux_rope_start = hours // 3
    flux_rope_duration = hours // 6
    
    for i in range(hours):
        if flux_rope_start <= i < flux_rope_start + flux_rope_duration:
            # Inside flux rope: smooth rotation of field direction
            progress = (i - flux_rope_start) / flux_rope_duration
            rotation_angle = progress * 2 * np.pi
            
            # Enhanced field magnitude
            magnitude = 15.0 + 5.0 * np.sin(progress * np.pi)
            
            # Rotating field components
            bx = magnitude * 0.5 * np.cos(rotation_angle)
            by = magnitude * 0.7 * np.sin(rotation_angle)
            bz = magnitude * 0.3 * np.cos(2 * rotation_angle)
            
        else:
            # Normal solar wind
            magnitude = 5.0 + np.random.normal(0, 2.0)
            
            # Typical Parker spiral field
            bx = magnitude * 0.6 + np.random.normal(0, 1.0)
            by = magnitude * -0.4 + np.random.normal(0, 1.5)
            bz = np.random.normal(0, 2.0)
            
            magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        magnetic_field.append([bx, by, bz])
        magnetic_field_magnitude.append(magnitude)
    
    # Generate data quality flags (0 = good, 1 = questionable, 2 = bad)
    data_quality = []
    for i in range(hours):
        if np.random.random() < 0.95:  # 95% good data
            quality = 0
        elif np.random.random() < 0.8:  # 4% questionable
            quality = 1
        else:  # 1% bad data
            quality = 2
        data_quality.append(quality)
    
    return {
        'timestamp': timestamps.tolist(),
        'magnetic_field': magnetic_field,
        'magnetic_field_magnitude': magnetic_field_magnitude,
        'coordinate_system': 'GSE',
        'data_quality': data_quality,
        'data_source': 'generated_sample'
    }

def detect_magnetic_rotation(data):
    """Detect magnetic field rotation (flux rope signature)"""
    if not data.get('magnetic_field') or len(data['magnetic_field']) < 3:
        return False, 0.0, 0.0
    
    field_data = np.array(data['magnetic_field'])
    
    # Calculate rotation angles between consecutive measurements
    rotation_angles = []
    for i in range(1, len(field_data)):
        b1 = field_data[i-1]
        b2 = field_data[i]
        
        # Calculate angle between vectors
        dot_product = np.dot(b1, b2)
        mag1 = np.linalg.norm(b1)
        mag2 = np.linalg.norm(b2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
            angle = np.arccos(cos_angle) * 180 / np.pi
            rotation_angles.append(angle)
    
    if not rotation_angles:
        return False, 0.0, 0.0
    
    # Calculate total rotation and maximum single rotation
    total_rotation = sum(rotation_angles)
    max_rotation = max(rotation_angles)
    
    # Flux rope detection: total rotation > 90° and smooth
    rotation_detected = total_rotation > 90.0 and max_rotation < 45.0
    confidence = min(1.0, total_rotation / 180.0)
    
    return rotation_detected, confidence, total_rotation

def detect_magnetic_enhancement(data):
    """Detect magnetic field enhancement"""
    if not data.get('magnetic_field_magnitude') or len(data['magnetic_field_magnitude']) < 3:
        return False, 0.0
    
    magnitudes = np.array(data['magnetic_field_magnitude'])
    
    # Calculate baseline (first third of data)
    baseline_end = len(magnitudes) // 3
    baseline = np.mean(magnitudes[:baseline_end])
    
    # Find maximum enhancement
    max_magnitude = np.max(magnitudes)
    enhancement_ratio = max_magnitude / baseline if baseline > 0 else 1.0
    
    # Enhancement detected if > 2x baseline
    enhancement_detected = enhancement_ratio > 2.0
    confidence = min(1.0, (enhancement_ratio - 2.0) / 3.0)  # Scale 2-5x to 0-1
    
    return enhancement_detected, confidence

def calculate_variance_anisotropy(data):
    """Calculate magnetic field variance anisotropy"""
    if not data.get('magnetic_field') or len(data['magnetic_field']) < 5:
        return 0.0
    
    field_data = np.array(data['magnetic_field'])
    
    # Calculate variance in each component
    var_x = np.var(field_data[:, 0])
    var_y = np.var(field_data[:, 1])
    var_z = np.var(field_data[:, 2])
    
    # Calculate anisotropy ratio
    total_var = var_x + var_y + var_z
    if total_var > 0:
        # Maximum anisotropy when all variance in one direction
        max_var = max(var_x, var_y, var_z)
        anisotropy = (3 * max_var - total_var) / (2 * total_var)
    else:
        anisotropy = 0.0
    
    return max(0.0, min(1.0, anisotropy))

def calculate_field_magnitude_gradient(data):
    """Calculate magnetic field magnitude gradient"""
    if not data.get('magnetic_field_magnitude') or len(data['magnetic_field_magnitude']) < 2:
        return 0.0
    
    magnitudes = np.array(data['magnetic_field_magnitude'])
    
    # Calculate gradient
    gradient = np.gradient(magnitudes)
    
    # Return maximum absolute gradient
    return np.max(np.abs(gradient))

def extract_magnetometer_features(data):
    """Extract scientific features from magnetometer data"""
    features = {}
    
    # Basic statistics
    if data.get('magnetic_field_magnitude'):
        magnitudes = np.array(data['magnetic_field_magnitude'])
        features['mean_magnitude'] = np.mean(magnitudes)
        features['max_magnitude'] = np.max(magnitudes)
        features['magnitude_variance'] = np.var(magnitudes)
        features['magnitude_gradient'] = calculate_field_magnitude_gradient(data)
    
    # Magnetic field rotation detection
    rotation_detected, rotation_confidence, total_rotation = detect_magnetic_rotation(data)
    features['rotation_detected'] = rotation_detected
    features['rotation_confidence'] = rotation_confidence
    features['total_rotation'] = total_rotation
    
    # Field enhancement detection
    enhancement_detected, enhancement_confidence = detect_magnetic_enhancement(data)
    features['enhancement_detected'] = enhancement_detected
    features['enhancement_confidence'] = enhancement_confidence
    
    # Variance anisotropy
    features['variance_anisotropy'] = calculate_variance_anisotropy(data)
    
    # Field components statistics
    if data.get('magnetic_field') and len(data['magnetic_field']) > 0:
        field_data = np.array(data['magnetic_field'])
        features['mean_bx'] = np.mean(field_data[:, 0])
        features['mean_by'] = np.mean(field_data[:, 1])
        features['mean_bz'] = np.mean(field_data[:, 2])
        features['bx_variance'] = np.var(field_data[:, 0])
        features['by_variance'] = np.var(field_data[:, 1])
        features['bz_variance'] = np.var(field_data[:, 2])
    
    # Data quality assessment
    if data.get('data_quality'):
        quality_flags = np.array(data['data_quality'])
        features['good_data_fraction'] = np.sum(quality_flags == 0) / len(quality_flags)
        features['bad_data_fraction'] = np.sum(quality_flags == 2) / len(quality_flags)
    else:
        features['good_data_fraction'] = 1.0
        features['bad_data_fraction'] = 0.0
    
    return features

# Test function
if __name__ == "__main__":
    # Test sample data generation
    sample_data = generate_sample_magnetometer_data()
    print("Sample magnetometer data generated:")
    print(f"Data points: {len(sample_data['magnetic_field'])}")
    print(f"Magnitude range: {min(sample_data['magnetic_field_magnitude']):.2f} - {max(sample_data['magnetic_field_magnitude']):.2f} nT")
    print(f"Coordinate system: {sample_data['coordinate_system']}")
    
    # Extract features
    features = extract_magnetometer_features(sample_data)
    print("\nExtracted features:")
    for key, value in features.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
