"""
Prediction and Forecasting Module for TITANUS CME Detection
Implements time-series prediction and future state forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class SpaceWeatherPredictor:
    def __init__(self):
        self.prediction_horizon = 24  # hours
        self.min_data_points = 5
        
    def generate_predictions(self, data_sources):
        """Generate future predictions from available data sources"""
        predictions = {}
        
        try:
            # Extract time series from each data source
            time_series_data = self.extract_time_series(data_sources)
            
            # Generate predictions for each parameter
            for parameter, values in time_series_data.items():
                if len(values) >= self.min_data_points:
                    predicted_values = self.predict_time_series(values, self.prediction_horizon)
                    predictions[parameter] = predicted_values
                else:
                    # Use simple extrapolation for insufficient data
                    predictions[parameter] = self.simple_extrapolation(values, self.prediction_horizon)
            
            # Add derived predictions
            predictions.update(self.generate_derived_predictions(predictions))
            
            return predictions
            
        except Exception as e:
            print(f"Prediction generation failed: {e}")
            return self.generate_default_predictions()
    
    def extract_time_series(self, data_sources):
        """Extract time series data from all sources"""
        time_series = {}
        
        # SWIS data
        if data_sources.get('swis'):
            swis = data_sources['swis']
            if swis.get('speed'):
                time_series['solar_wind_speed'] = swis['speed']
            if swis.get('density'):
                time_series['proton_density'] = swis['density']
            if swis.get('temperature'):
                time_series['temperature'] = swis['temperature']
        
        # SOLERIOX data
        if data_sources.get('soleriox'):
            soleriox = data_sources['soleriox']
            if soleriox.get('ion_flux'):
                time_series['ion_flux'] = soleriox['ion_flux']
            if soleriox.get('electron_flux'):
                time_series['electron_flux'] = soleriox['electron_flux']
        
        # Magnetometer data
        if data_sources.get('magnetometer'):
            mag = data_sources['magnetometer']
            if mag.get('magnetic_field_magnitude'):
                time_series['magnetic_field_magnitude'] = mag['magnetic_field_magnitude']
            if mag.get('magnetic_field'):
                # Extract components
                mag_field = mag['magnetic_field']
                if len(mag_field) > 0 and len(mag_field[0]) >= 3:
                    time_series['magnetic_field_x'] = [b[0] for b in mag_field]
                    time_series['magnetic_field_y'] = [b[1] for b in mag_field]
                    time_series['magnetic_field_z'] = [b[2] for b in mag_field]
        
        return time_series
    
    def predict_time_series(self, values, horizon):
        """Predict future values using multiple methods and ensemble"""
        values = np.array(values)
        
        # Method 1: Linear trend extrapolation
        linear_pred = self.linear_extrapolation(values, horizon)
        
        # Method 2: Exponential smoothing
        exp_pred = self.exponential_smoothing(values, horizon)
        
        # Method 3: Polynomial extrapolation
        poly_pred = self.polynomial_extrapolation(values, horizon)
        
        # Method 4: Seasonal decomposition (if enough data)
        if len(values) >= 12:
            seasonal_pred = self.seasonal_prediction(values, horizon)
        else:
            seasonal_pred = linear_pred
        
        # Ensemble prediction (weighted average)
        weights = [0.3, 0.25, 0.25, 0.2]  # Adjust based on method reliability
        ensemble_pred = []
        
        for i in range(horizon):
            pred_value = (weights[0] * linear_pred[i] +
                         weights[1] * exp_pred[i] +
                         weights[2] * poly_pred[i] +
                         weights[3] * seasonal_pred[i])
            ensemble_pred.append(pred_value)
        
        # Apply physical constraints
        ensemble_pred = self.apply_physical_constraints(ensemble_pred, values)
        
        return ensemble_pred
    
    def linear_extrapolation(self, values, horizon):
        """Linear trend extrapolation"""
        n = len(values)
        x = np.arange(n)
        
        # Fit linear regression
        reg = LinearRegression().fit(x.reshape(-1, 1), values)
        
        # Predict future values
        future_x = np.arange(n, n + horizon)
        predictions = reg.predict(future_x.reshape(-1, 1))
        
        return predictions.tolist()
    
    def exponential_smoothing(self, values, horizon, alpha=0.3):
        """Exponential smoothing prediction"""
        if len(values) == 0:
            return [0.0] * horizon
        
        # Calculate exponentially smoothed values
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
        
        # Calculate trend
        if len(values) >= 2:
            trend = smoothed[-1] - smoothed[-2]
        else:
            trend = 0
        
        # Extrapolate
        predictions = []
        last_value = smoothed[-1]
        
        for i in range(horizon):
            pred_value = last_value + trend * (i + 1)
            predictions.append(pred_value)
        
        return predictions
    
    def polynomial_extrapolation(self, values, horizon, degree=2):
        """Polynomial extrapolation"""
        n = len(values)
        
        # Limit degree to avoid overfitting
        max_degree = min(degree, n - 1, 3)
        
        if max_degree < 1:
            return [values[-1]] * horizon
        
        x = np.arange(n)
        
        # Fit polynomial
        coeffs = np.polyfit(x, values, max_degree)
        poly = np.poly1d(coeffs)
        
        # Predict future values
        future_x = np.arange(n, n + horizon)
        predictions = poly(future_x)
        
        return predictions.tolist()
    
    def seasonal_prediction(self, values, horizon):
        """Seasonal decomposition and prediction"""
        try:
            # Simple seasonal decomposition
            n = len(values)
            
            # Estimate period (assume daily or semi-daily patterns)
            possible_periods = [6, 8, 12, 24]  # hours
            best_period = self.estimate_period(values, possible_periods)
            
            if best_period and best_period < n // 2:
                # Extract seasonal component
                seasonal = self.extract_seasonal_component(values, best_period)
                
                # Detrend
                detrended = values - seasonal[:len(values)]
                
                # Predict trend
                trend_pred = self.linear_extrapolation(detrended, horizon)
                
                # Add seasonal component
                predictions = []
                for i in range(horizon):
                    seasonal_idx = (len(values) + i) % best_period
                    if seasonal_idx < len(seasonal):
                        seasonal_component = seasonal[seasonal_idx]
                    else:
                        seasonal_component = 0
                    
                    predictions.append(trend_pred[i] + seasonal_component)
                
                return predictions
            else:
                # Fall back to linear extrapolation
                return self.linear_extrapolation(values, horizon)
                
        except Exception as e:
            print(f"Seasonal prediction failed: {e}")
            return self.linear_extrapolation(values, horizon)
    
    def estimate_period(self, values, possible_periods):
        """Estimate dominant period in time series"""
        best_period = None
        max_correlation = 0
        
        for period in possible_periods:
            if period < len(values) // 2:
                # Calculate autocorrelation at this lag
                correlation = self.autocorrelation(values, period)
                if correlation > max_correlation:
                    max_correlation = correlation
                    best_period = period
        
        return best_period if max_correlation > 0.3 else None
    
    def autocorrelation(self, values, lag):
        """Calculate autocorrelation at given lag"""
        if lag >= len(values):
            return 0
        
        n = len(values) - lag
        x1 = values[:n]
        x2 = values[lag:lag+n]
        
        correlation = np.corrcoef(x1, x2)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def extract_seasonal_component(self, values, period):
        """Extract seasonal component with given period"""
        n = len(values)
        seasonal = np.zeros(period)
        counts = np.zeros(period)
        
        # Average values at each phase
        for i in range(n):
            phase = i % period
            seasonal[phase] += values[i]
            counts[phase] += 1
        
        # Normalize
        for i in range(period):
            if counts[i] > 0:
                seasonal[i] /= counts[i]
        
        # Remove mean to get seasonal component
        seasonal -= np.mean(seasonal)
        
        return seasonal
    
    def apply_physical_constraints(self, predictions, historical_values):
        """Apply physical constraints to predictions"""
        if not historical_values:
            return predictions
        
        # Calculate reasonable bounds based on historical data
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        min_val = np.min(historical_values)
        max_val = np.max(historical_values)
        
        # Allow predictions to vary within reasonable bounds
        lower_bound = max(0, min_val - 2 * std_val)  # Don't go negative for physical quantities
        upper_bound = max_val + 3 * std_val
        
        # Apply bounds
        constrained_predictions = []
        for pred in predictions:
            constrained_pred = max(lower_bound, min(upper_bound, pred))
            constrained_predictions.append(constrained_pred)
        
        return constrained_predictions
    
    def simple_extrapolation(self, values, horizon):
        """Simple extrapolation for insufficient data"""
        if not values:
            return [0.0] * horizon
        
        if len(values) == 1:
            # Constant prediction
            return [values[0]] * horizon
        
        # Linear extrapolation from last two points
        if len(values) >= 2:
            slope = values[-1] - values[-2]
            last_value = values[-1]
            
            predictions = []
            for i in range(horizon):
                pred = last_value + slope * (i + 1)
                predictions.append(pred)
            
            return predictions
        
        return [values[-1]] * horizon
    
    def generate_derived_predictions(self, predictions):
        """Generate predictions for derived parameters"""
        derived = {}
        
        # Dynamic pressure from speed and density
        if 'solar_wind_speed' in predictions and 'proton_density' in predictions:
            speed_pred = predictions['solar_wind_speed']
            density_pred = predictions['proton_density']
            
            dynamic_pressure = []
            for s, d in zip(speed_pred, density_pred):
                # P_dyn = 1.67e-6 * n * v^2 (nPa)
                pressure = 1.67e-6 * d * s**2
                dynamic_pressure.append(pressure)
            
            derived['dynamic_pressure'] = dynamic_pressure
        
        # Alfvén speed from magnetic field and density
        if 'magnetic_field_magnitude' in predictions and 'proton_density' in predictions:
            b_pred = predictions['magnetic_field_magnitude']
            density_pred = predictions['proton_density']
            
            alfven_speed = []
            for b, d in zip(b_pred, density_pred):
                # V_A = B / sqrt(mu_0 * rho) ~ 21.8 * B / sqrt(n) (km/s)
                v_a = 21.8 * b / np.sqrt(d) if d > 0 else 0
                alfven_speed.append(v_a)
            
            derived['alfven_speed'] = alfven_speed
        
        # Plasma beta
        if ('temperature' in predictions and 
            'magnetic_field_magnitude' in predictions and 
            'proton_density' in predictions):
            
            temp_pred = predictions['temperature']
            b_pred = predictions['magnetic_field_magnitude']
            density_pred = predictions['proton_density']
            
            beta = []
            for t, b, d in zip(temp_pred, b_pred, density_pred):
                # beta = (n * k * T) / (B^2 / 2*mu_0)
                if b > 0:
                    plasma_beta = 4.03e-11 * d * t / (b**2)
                    beta.append(plasma_beta)
                else:
                    beta.append(0)
            
            derived['plasma_beta'] = beta
        
        return derived
    
    def generate_default_predictions(self):
        """Generate default predictions when data is insufficient"""
        horizon = self.prediction_horizon
        
        # Default solar wind conditions
        default_predictions = {
            'solar_wind_speed': [400 + np.random.normal(0, 20) for _ in range(horizon)],
            'proton_density': [5 + np.random.normal(0, 1) for _ in range(horizon)],
            'temperature': [100000 + np.random.normal(0, 10000) for _ in range(horizon)],
            'magnetic_field_magnitude': [6 + np.random.normal(0, 2) for _ in range(horizon)],
            'dynamic_pressure': [2 + np.random.normal(0, 0.5) for _ in range(horizon)]
        }
        
        # Ensure positive values
        for key, values in default_predictions.items():
            default_predictions[key] = [max(0.1, v) for v in values]
        
        return default_predictions
    
    def calculate_prediction_uncertainty(self, predictions, historical_values):
        """Calculate uncertainty bounds for predictions"""
        if not historical_values:
            return {}
        
        uncertainty = {}
        historical_std = np.std(historical_values)
        
        for param, pred_values in predictions.items():
            # Uncertainty grows with prediction horizon
            uncertainty_bounds = []
            
            for i, pred in enumerate(pred_values):
                # Uncertainty increases with time
                time_factor = 1 + 0.1 * i  # 10% increase per hour
                bound = historical_std * time_factor
                
                uncertainty_bounds.append({
                    'prediction': pred,
                    'lower_bound': pred - bound,
                    'upper_bound': pred + bound,
                    'uncertainty': bound
                })
            
            uncertainty[param] = uncertainty_bounds
        
        return uncertainty

def generate_predictions(data_sources):
    """Main function for external use"""
    predictor = SpaceWeatherPredictor()
    return predictor.generate_predictions(data_sources)

def predict_cme_arrival_time(cme_detection_time, solar_wind_speed):
    """Predict CME arrival time at Earth"""
    try:
        # Typical CME-Earth distance: 150 million km (1 AU)
        distance_km = 1.5e8
        
        # Average CME speed (km/s)
        if isinstance(solar_wind_speed, list):
            avg_speed = np.mean(solar_wind_speed)
        else:
            avg_speed = solar_wind_speed
        
        # CME speed is typically 1.2-2x faster than solar wind
        cme_speed = avg_speed * 1.5
        
        # Calculate travel time in seconds
        travel_time_seconds = distance_km / cme_speed
        
        # Convert to hours
        travel_time_hours = travel_time_seconds / 3600
        
        # Calculate arrival time
        arrival_time = cme_detection_time + timedelta(hours=travel_time_hours)
        
        return {
            'arrival_time': arrival_time,
            'travel_time_hours': travel_time_hours,
            'estimated_cme_speed': cme_speed,
            'uncertainty_hours': travel_time_hours * 0.3  # ±30% uncertainty
        }
        
    except Exception as e:
        print(f"CME arrival time prediction failed: {e}")
        return {
            'arrival_time': cme_detection_time + timedelta(hours=36),  # Default 36 hours
            'travel_time_hours': 36,
            'estimated_cme_speed': 500,
            'uncertainty_hours': 12
        }

def assess_geomagnetic_impact(predictions):
    """Assess potential geomagnetic impact from predictions"""
    try:
        impact_score = 0
        impact_factors = []
        
        # Solar wind speed impact
        if 'solar_wind_speed' in predictions:
            max_speed = max(predictions['solar_wind_speed'])
            if max_speed > 600:
                speed_impact = min(5, (max_speed - 400) / 100)
                impact_score += speed_impact
                impact_factors.append(f"Enhanced solar wind speed: {max_speed:.0f} km/s")
        
        # Dynamic pressure impact
        if 'dynamic_pressure' in predictions:
            max_pressure = max(predictions['dynamic_pressure'])
            if max_pressure > 5:
                pressure_impact = min(3, (max_pressure - 2) / 2)
                impact_score += pressure_impact
                impact_factors.append(f"Enhanced dynamic pressure: {max_pressure:.1f} nPa")
        
        # Magnetic field impact
        if 'magnetic_field_magnitude' in predictions:
            max_field = max(predictions['magnetic_field_magnitude'])
            if max_field > 15:
                field_impact = min(2, (max_field - 10) / 10)
                impact_score += field_impact
                impact_factors.append(f"Enhanced magnetic field: {max_field:.1f} nT")
        
        # Determine impact level
        if impact_score > 7:
            impact_level = "SEVERE"
        elif impact_score > 5:
            impact_level = "MODERATE"
        elif impact_score > 2:
            impact_level = "MINOR"
        else:
            impact_level = "MINIMAL"
        
        return {
            'impact_level': impact_level,
            'impact_score': impact_score,
            'impact_factors': impact_factors,
            'kp_estimate': min(9, max(1, impact_score * 1.2)),
            'storm_probability': min(100, impact_score * 12)  # Percentage
        }
        
    except Exception as e:
        print(f"Geomagnetic impact assessment failed: {e}")
        return {
            'impact_level': "UNKNOWN",
            'impact_score': 0,
            'impact_factors': [],
            'kp_estimate': 3,
            'storm_probability': 0
        }

if __name__ == "__main__":
    # Test the prediction system
    print("Testing TITANUS prediction system...")
    
    # Create sample data sources
    sample_data = {
        'swis': {
            'speed': [400, 450, 500, 600, 650, 620],
            'density': [5, 6, 8, 12, 15, 10],
            'temperature': [100000, 95000, 80000, 70000, 75000, 85000]
        },
        'magnetometer': {
            'magnetic_field_magnitude': [6, 8, 12, 18, 20, 16]
        },
        'soleriox': {
            'ion_flux': [1e5, 2e5, 5e5, 1e6, 8e5, 6e5]
        }
    }
    
    # Generate predictions
    predictor = SpaceWeatherPredictor()
    predictions = predictor.generate_predictions(sample_data)
    
    print(f"Generated predictions for {len(predictions)} parameters:")
    for param, values in predictions.items():
        print(f"  {param}: {len(values)} hours, range {min(values):.2f} - {max(values):.2f}")
    
    # Test CME arrival time prediction
    cme_time = datetime.now()
    arrival_info = predict_cme_arrival_time(cme_time, 650)
    print(f"\nCME arrival prediction:")
    print(f"  Arrival time: {arrival_info['arrival_time']}")
    print(f"  Travel time: {arrival_info['travel_time_hours']:.1f} hours")
    
    # Test geomagnetic impact assessment
    impact = assess_geomagnetic_impact(predictions)
    print(f"\nGeomagnetic impact assessment:")
    print(f"  Impact level: {impact['impact_level']}")
    print(f"  Estimated Kp: {impact['kp_estimate']:.1f}")
    print(f"  Storm probability: {impact['storm_probability']:.0f}%")
    
    print("\nPrediction system test completed successfully!")
