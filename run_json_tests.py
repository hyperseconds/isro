#!/usr/bin/env python3
"""
Non-interactive JSON test runner for TITANUS CME Prediction System
"""

import json
import os
import subprocess
from datetime import datetime, timedelta

def create_test_scenarios():
    """Create different test scenarios for CME prediction"""
    
    scenarios = {
        # Scenario 1: Strong CME Event (Multiple thresholds triggered)
        "strong_cme": {
            "timestamp": "2025-07-15T15:00:00Z",
            "sources": ["swis", "soleriox", "magnetometer"],
            "features": {
                "solar_wind_speed": 750.0,      # > 600 threshold
                "proton_density": 15.5,         # > 10 threshold
                "temperature": 45000.0,         # < 50000 threshold (temperature depression)
                "ion_flux": 3500000.0,          # > 1e6 threshold
                "electron_flux": 25000000.0,    # > 1e7 threshold
                "magnetic_field_x": 12.8,
                "magnetic_field_y": -18.5,
                "magnetic_field_z": 22.3,
                "magnetic_field_magnitude": 32.1,  # > 20 threshold
                "dynamic_pressure": 12.8        # > 5 threshold
            }
        },
        
        # Scenario 2: Moderate CME Event
        "moderate_cme": {
            "timestamp": "2025-07-15T16:00:00Z",
            "sources": ["swis", "soleriox", "magnetometer"],
            "features": {
                "solar_wind_speed": 620.0,      # Just above threshold
                "proton_density": 11.2,         # Slightly above threshold
                "temperature": 52000.0,         # Normal temperature
                "ion_flux": 1200000.0,          # Above threshold
                "electron_flux": 8500000.0,     # Below electron threshold
                "magnetic_field_x": 8.2,
                "magnetic_field_y": -10.1,
                "magnetic_field_z": 12.7,
                "magnetic_field_magnitude": 18.5,  # Below magnetic threshold
                "dynamic_pressure": 6.1         # Above pressure threshold
            }
        },
        
        # Scenario 3: Quiet Space Weather (No CME)
        "quiet_conditions": {
            "timestamp": "2025-07-15T17:00:00Z",
            "sources": ["swis", "soleriox", "magnetometer"],
            "features": {
                "solar_wind_speed": 420.0,      # Below threshold
                "proton_density": 5.8,          # Below threshold
                "temperature": 65000.0,         # Normal temperature
                "ion_flux": 450000.0,           # Below threshold
                "electron_flux": 3200000.0,     # Below threshold
                "magnetic_field_x": 5.2,
                "magnetic_field_y": -3.8,
                "magnetic_field_z": 7.1,
                "magnetic_field_magnitude": 12.3,  # Below threshold
                "dynamic_pressure": 2.8         # Below threshold
            }
        },
        
        # Scenario 4: Borderline CME Event (Exactly at thresholds)
        "borderline_cme": {
            "timestamp": "2025-07-15T18:00:00Z",
            "sources": ["swis", "soleriox", "magnetometer"],
            "features": {
                "solar_wind_speed": 600.0,      # Exactly at threshold
                "proton_density": 10.0,         # Exactly at threshold
                "temperature": 50000.0,         # Exactly at threshold
                "ion_flux": 1000000.0,          # Exactly at threshold
                "electron_flux": 10000000.0,    # Exactly at threshold
                "magnetic_field_x": 6.0,
                "magnetic_field_y": -8.0,
                "magnetic_field_z": 10.0,
                "magnetic_field_magnitude": 20.0,  # Exactly at threshold
                "dynamic_pressure": 5.0         # Exactly at threshold
            }
        }
    }
    
    return scenarios

def test_scenario(scenario_name, scenario_data):
    """Test a specific scenario"""
    print(f"\n{'='*50}")
    print(f"Testing Scenario: {scenario_name.upper()}")
    print(f"{'='*50}")
    
    # Write test data to input JSON file
    with open('fused/fused_input.json', 'w') as f:
        json.dump(scenario_data, f, indent=2)
    
    # Run the C prediction engine
    try:
        result = subprocess.run(['./c_core/titanus_predictor'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ C Engine executed successfully")
            
            # Read the prediction output
            if os.path.exists('fused/prediction_output.json'):
                with open('fused/prediction_output.json', 'r') as f:
                    prediction = json.load(f)
                
                print_prediction_analysis(scenario_data, prediction)
            else:
                print("❌ No prediction output file generated")
        else:
            print(f"❌ C Engine failed with error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ C Engine timed out")
    except Exception as e:
        print(f"❌ Error running C Engine: {e}")

def print_prediction_analysis(input_data, prediction):
    """Print detailed analysis of the prediction results"""
    
    print(f"\n🔍 INPUT DATA ANALYSIS:")
    features = input_data['features']
    
    print(f"  Solar Wind Speed: {features['solar_wind_speed']:.1f} km/s (threshold: 600)")
    print(f"  Proton Density: {features['proton_density']:.1f} cm⁻³ (threshold: 10)")
    print(f"  Temperature: {features['temperature']:.0f} K (threshold: 50000)")
    print(f"  Ion Flux: {features['ion_flux']:.0f} particles/cm²/s (threshold: 1e6)")
    print(f"  Electron Flux: {features['electron_flux']:.0f} particles/cm²/s (threshold: 1e7)")
    print(f"  Magnetic Field: {features['magnetic_field_magnitude']:.1f} nT (threshold: 20)")
    print(f"  Dynamic Pressure: {features['dynamic_pressure']:.1f} nPa (threshold: 5)")
    
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"  CME Detected: {'YES' if prediction['cme_detected'] else 'NO'}")
    print(f"  Confidence: {prediction['confidence']:.1%}")
    print(f"  Method: {prediction['prediction_method']}")
    
    if 'detection_analysis' in prediction:
        analysis = prediction['detection_analysis']
        print(f"  Detection Flags: {analysis['detection_flags']}/{analysis['required_flags']}")
        print(f"  Triggered Thresholds: {', '.join(analysis['triggered_thresholds'])}")
    
    print(f"\n📊 24-HOUR FORECAST:")
    future = prediction['future_predictions']
    print(f"  Solar Wind Speed: {future['solar_wind_speed'][0]:.0f} → {future['solar_wind_speed'][23]:.0f} km/s")
    print(f"  Proton Density: {future['proton_density'][0]:.1f} → {future['proton_density'][23]:.1f} cm⁻³")
    print(f"  Magnetic Field: {future['magnetic_field_magnitude'][0]:.1f} → {future['magnetic_field_magnitude'][23]:.1f} nT")
    print(f"  Dynamic Pressure: {future['dynamic_pressure'][0]:.1f} → {future['dynamic_pressure'][23]:.1f} nPa")
    
    if 'geomagnetic_forecast' in prediction:
        geo = prediction['geomagnetic_forecast']
        print(f"\n🌍 GEOMAGNETIC IMPACT:")
        print(f"  Impact Level: {geo['impact_level']}")
        print(f"  Kp Index Estimate: {geo['kp_estimate']}")
        print(f"  Storm Probability: {geo['storm_probability']:.1f}%")
        print(f"  Estimated Arrival: {geo['arrival_time_estimate']}")

def main():
    """Main test function"""
    print("🚀 TITANUS CME Prediction System - JSON Test Mode")
    print("=" * 60)
    
    # Check if C engine is compiled
    if not os.path.exists('c_core/titanus_predictor'):
        print("❌ C engine not found. Compiling...")
        try:
            subprocess.run(['make', '-C', 'c_core', 'all'], check=True)
            print("✅ C engine compiled successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to compile C engine")
            return
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Test each scenario
    for scenario_name, scenario_data in scenarios.items():
        test_scenario(scenario_name, scenario_data)
    
    print(f"\n🎯 All tests completed!")
    print(f"You can modify the scenarios in this script to test different conditions.")
    print(f"The JSON files are saved in fused/ directory for inspection.")

if __name__ == "__main__":
    main()