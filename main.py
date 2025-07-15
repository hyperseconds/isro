#!/usr/bin/env python3
"""
TITANUS CME Detection and Prediction System
Main GUI Application with Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import subprocess
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import psycopg2
from datetime import datetime, timedelta

# Import local modules
from parsers import swis_parser, soleriox_parser, magnetometer_parser
from utils import db_logger, report_generator, email_alert, predictor
from training import threshold_trainer
from models import init_database

class TitanusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TITANUS - CME Detection & Prediction System")
        self.root.geometry("1200x800")
        
        # Initialize database
        try:
            self.db_conn = init_database()
            print("Database connected successfully")
        except Exception as e:
            print(f"Database connection failed: {e}")
            self.db_conn = None
            
        # Initialize variables
        self.data_sources = {
            'swis': None,
            'soleriox': None,
            'magnetometer': None
        }
        self.fused_data = None
        self.prediction_results = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="TITANUS CME Detection & Prediction System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Data Input Section
        data_frame = ttk.LabelFrame(main_frame, text="Data Input", padding="10")
        data_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # CACTUS CSV Upload (for training comparison)
        cactus_frame = ttk.Frame(data_frame)
        cactus_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(cactus_frame, text="CACTUS CME Catalog (CSV):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Button(cactus_frame, text="Upload CACTUS CSV", 
                  command=self.load_cactus_csv).grid(row=0, column=1, padx=5)
        
        self.cactus_status = ttk.Label(cactus_frame, text="Not Loaded", foreground="red")
        self.cactus_status.grid(row=0, column=2, padx=5)
        
        # JSON Input Section
        json_frame = ttk.LabelFrame(data_frame, text="Space Weather Data (JSON Input)", padding="10")
        json_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        data_frame.rowconfigure(1, weight=1)
        
        # JSON input area
        json_input_frame = ttk.Frame(json_frame)
        json_input_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        json_frame.rowconfigure(0, weight=1)
        json_frame.columnconfigure(0, weight=1)
        
        ttk.Label(json_input_frame, text="Enter space weather parameters (JSON format):").grid(row=0, column=0, sticky=tk.W)
        
        self.json_text = tk.Text(json_input_frame, height=12, width=80)
        json_scrollbar = ttk.Scrollbar(json_input_frame, orient="vertical", command=self.json_text.yview)
        self.json_text.configure(yscrollcommand=json_scrollbar.set)
        
        self.json_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        json_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        json_input_frame.columnconfigure(0, weight=1)
        json_input_frame.rowconfigure(1, weight=1)
        
        # JSON control buttons
        json_control_frame = ttk.Frame(json_frame)
        json_control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        ttk.Button(json_control_frame, text="Strong CME Example", 
                  command=lambda: self.load_sample_json('strong_cme')).grid(row=0, column=0, padx=5)
        ttk.Button(json_control_frame, text="Moderate CME Example", 
                  command=lambda: self.load_sample_json('moderate_cme')).grid(row=0, column=1, padx=5)
        ttk.Button(json_control_frame, text="Quiet Conditions", 
                  command=lambda: self.load_sample_json('quiet')).grid(row=0, column=2, padx=5)
        ttk.Button(json_control_frame, text="Clear JSON", 
                  command=self.clear_json).grid(row=0, column=3, padx=5)
        
        # Status indicator
        self.json_status = ttk.Label(json_control_frame, text="No Data", foreground="red")
        self.json_status.grid(row=0, column=4, padx=10)
        
        # Control Buttons Section
        control_frame = ttk.LabelFrame(main_frame, text="Operations", padding="10")
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="🚀 Predict CME", 
                  command=self.predict_from_json).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="🎯 Train Thresholds", 
                  command=self.train_thresholds).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="💾 Log to DB", 
                  command=self.log_to_database).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="📄 Generate PDF", 
                  command=self.generate_pdf_report).grid(row=0, column=3, padx=5)
        
        # Prediction Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.results_text = tk.Text(results_frame, height=8, width=80)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Plot Frame
        plot_frame = ttk.LabelFrame(main_frame, text="Data Visualization", padding="10")
        plot_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Initialize matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
    def load_sample_json(self, scenario='strong_cme'):
        """Load sample JSON data for testing different scenarios"""
        
        scenarios = {
            'strong_cme': {
                "timestamp": "2025-07-15T15:00:00Z",
                "scenario": "Strong CME Event",
                "description": "Multiple thresholds triggered - high probability CME",
                "swis_data": {
                    "solar_wind_speed": 750.0,     # > 600 threshold
                    "proton_density": 15.5,        # > 10 threshold  
                    "temperature": 45000.0         # < 50000 threshold (depression)
                },
                "soleriox_data": {
                    "ion_flux": 3500000.0,         # > 1e6 threshold
                    "electron_flux": 25000000.0    # > 1e7 threshold
                },
                "magnetometer_data": {
                    "magnetic_field_x": 12.8,
                    "magnetic_field_y": -18.5,
                    "magnetic_field_z": 22.3,
                    "magnetic_field_magnitude": 32.1  # > 20 threshold
                },
                "derived_parameters": {
                    "dynamic_pressure": 12.8       # > 5 threshold
                }
            },
            'moderate_cme': {
                "timestamp": "2025-07-15T16:00:00Z",
                "scenario": "Moderate CME Event",
                "description": "Some thresholds triggered - possible CME",
                "swis_data": {
                    "solar_wind_speed": 620.0,     # Just above threshold
                    "proton_density": 11.2,        # Slightly above threshold
                    "temperature": 52000.0         # Normal temperature
                },
                "soleriox_data": {
                    "ion_flux": 1200000.0,         # Above threshold
                    "electron_flux": 8500000.0     # Below electron threshold
                },
                "magnetometer_data": {
                    "magnetic_field_x": 8.2,
                    "magnetic_field_y": -10.1,
                    "magnetic_field_z": 12.7,
                    "magnetic_field_magnitude": 18.5  # Below magnetic threshold
                },
                "derived_parameters": {
                    "dynamic_pressure": 6.1        # Above pressure threshold
                }
            },
            'quiet': {
                "timestamp": "2025-07-15T17:00:00Z",
                "scenario": "Quiet Space Weather",
                "description": "Normal conditions - no CME expected",
                "swis_data": {
                    "solar_wind_speed": 420.0,     # Below threshold
                    "proton_density": 5.8,         # Below threshold
                    "temperature": 65000.0         # Normal temperature
                },
                "soleriox_data": {
                    "ion_flux": 450000.0,          # Below threshold
                    "electron_flux": 3200000.0     # Below threshold
                },
                "magnetometer_data": {
                    "magnetic_field_x": 5.2,
                    "magnetic_field_y": -3.8,
                    "magnetic_field_z": 7.1,
                    "magnetic_field_magnitude": 12.3  # Below threshold
                },
                "derived_parameters": {
                    "dynamic_pressure": 2.8        # Below threshold
                }
            }
        }
        
        if scenario not in scenarios:
            scenario = 'strong_cme'
        
        data = scenarios[scenario]
        
        # Flatten the data into the required format
        flattened_data = {
            "timestamp": data["timestamp"],
            "scenario": data["scenario"],
            "description": data["description"],
            "features": {
                "solar_wind_speed": data["swis_data"]["solar_wind_speed"],
                "proton_density": data["swis_data"]["proton_density"],
                "temperature": data["swis_data"]["temperature"],
                "ion_flux": data["soleriox_data"]["ion_flux"],
                "electron_flux": data["soleriox_data"]["electron_flux"],
                "magnetic_field_x": data["magnetometer_data"]["magnetic_field_x"],
                "magnetic_field_y": data["magnetometer_data"]["magnetic_field_y"],
                "magnetic_field_z": data["magnetometer_data"]["magnetic_field_z"],
                "magnetic_field_magnitude": data["magnetometer_data"]["magnetic_field_magnitude"],
                "dynamic_pressure": data["derived_parameters"]["dynamic_pressure"]
            },
            "thresholds_info": {
                "solar_wind_speed": "CME if > 600 km/s",
                "proton_density": "CME if > 10 cm⁻³",
                "temperature": "CME if < 50000 K (depression)",
                "ion_flux": "CME if > 1e6 particles/cm²/s",
                "electron_flux": "CME if > 1e7 particles/cm²/s",
                "magnetic_field_magnitude": "CME if > 20 nT",
                "dynamic_pressure": "CME if > 5 nPa"
            }
        }
        
        self.json_text.delete(1.0, tk.END)
        self.json_text.insert(1.0, json.dumps(flattened_data, indent=2))
        self.json_status.config(text=f"Loaded: {data['scenario']}", foreground="green")
        self.update_results(f"Sample JSON data loaded: {data['scenario']}")
        self.update_results(f"Description: {data['description']}")
        
    def load_cactus_csv(self):
        """Load CACTUS CME catalog CSV for training/comparison"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select CACTUS CME Catalog CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Copy to data directory
            import shutil
            target_path = "data/CACTUS_events.csv"
            shutil.copy2(file_path, target_path)
            
            # Load and validate
            df = pd.read_csv(target_path)
            self.cactus_status.config(text=f"Loaded ({len(df)} events)", foreground="green")
            self.update_results(f"CACTUS catalog loaded: {len(df)} CME events")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CACTUS CSV: {str(e)}")
            self.update_results(f"CACTUS loading error: {str(e)}")
        
    def load_from_json(self):
        """Load data from JSON input"""
        try:
            json_content = self.json_text.get(1.0, tk.END).strip()
            if not json_content:
                messagebox.showwarning("Warning", "Please enter JSON data first")
                return False
            
            # Parse JSON
            json_data = json.loads(json_content)
            
            # Validate required fields
            if 'features' not in json_data:
                messagebox.showerror("Error", "JSON must contain 'features' field")
                return False
            
            features = json_data['features']
            required_fields = [
                'solar_wind_speed', 'proton_density', 'temperature',
                'ion_flux', 'electron_flux', 'magnetic_field_x',
                'magnetic_field_y', 'magnetic_field_z', 'magnetic_field_magnitude',
                'dynamic_pressure'
            ]
            
            missing_fields = [field for field in required_fields if field not in features]
            if missing_fields:
                messagebox.showerror("Error", f"Missing required fields: {', '.join(missing_fields)}")
                return False
            
            # Create fused data from JSON
            self.fused_data = {
                'timestamp': json_data.get('timestamp', datetime.now().isoformat()),
                'sources': ['json_input'],
                'features': features,
                'data_quality': {
                    'overall_quality': 100.0,
                    'json_input_quality': 100.0
                }
            }
            
            # Update status
            self.json_status.config(text="Data Loaded", foreground="green")
            
            scenario = json_data.get('scenario', 'Custom Data')
            self.update_results(f"JSON data loaded successfully: {scenario}")
            
            if 'description' in json_data:
                self.update_results(f"Description: {json_data['description']}")
            
            return True
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON format: {str(e)}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {str(e)}")
            return False
    
    def clear_json(self):
        """Clear JSON input area"""
        self.json_text.delete(1.0, tk.END)
        self.json_status.config(text="No Data", foreground="red")
        self.fused_data = None
        self.update_results("JSON input cleared")
    
    def predict_from_json(self):
        """Predict CME using JSON input data"""
        try:
            # First load from JSON if not already loaded
            if not self.fused_data:
                if not self.load_from_json():
                    return
            
            # Write fused data to JSON file for C engine
            with open('fused/fused_input.json', 'w') as f:
                json.dump(self.fused_data, f, indent=2)
            
            # Compile C program if needed
            if not self.compile_c_program():
                return
            
            # Run C prediction engine
            self.update_results("🚀 Running C prediction engine...")
            
            result = subprocess.run(['./c_core/titanus_predictor'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.update_results("✅ C prediction engine completed successfully")
                self.display_prediction_results()
                
                # Send alert if CME detected
                if (self.prediction_results and 
                    self.prediction_results.get('cme_detected', False)):
                    self.send_cme_alert()
                
                # Plot the results
                self.plot_json_data()
                
            else:
                error_msg = f"❌ C engine failed: {result.stderr}"
                messagebox.showerror("Error", error_msg)
                self.update_results(error_msg)
                
        except subprocess.TimeoutExpired:
            messagebox.showerror("Error", "C prediction engine timed out")
            self.update_results("⏱️ C prediction engine timed out")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_results(f"❌ Prediction error: {str(e)}")
    
    def plot_json_data(self):
        """Plot data from JSON input"""
        try:
            if not self.fused_data or not self.prediction_results:
                return
                
            # Clear previous plots
            for ax in self.axes.flat:
                ax.clear()
            
            features = self.fused_data['features']
            future_pred = self.prediction_results.get('future_predictions', {})
            
            # Plot 1: Solar Wind Speed
            ax1 = self.axes[0, 0]
            current_speed = features['solar_wind_speed']
            if 'solar_wind_speed' in future_pred:
                future_hours = list(range(1, 25))
                future_speeds = future_pred['solar_wind_speed']
                
                ax1.plot([0], [current_speed], 'ro', markersize=8, label='Current')
                ax1.plot(future_hours, future_speeds, 'b-', label='24h Prediction')
                ax1.axhline(y=600, color='r', linestyle='--', label='CME Threshold')
                ax1.set_title('Solar Wind Speed')
                ax1.set_xlabel('Hours from Now')
                ax1.set_ylabel('Speed (km/s)')
                ax1.legend()
                ax1.grid(True)
            
            # Plot 2: Proton Density
            ax2 = self.axes[0, 1]
            current_density = features['proton_density']
            if 'proton_density' in future_pred:
                future_density = future_pred['proton_density']
                
                ax2.plot([0], [current_density], 'ro', markersize=8, label='Current')
                ax2.plot(future_hours, future_density, 'g-', label='24h Prediction')
                ax2.axhline(y=10, color='r', linestyle='--', label='CME Threshold')
                ax2.set_title('Proton Density')
                ax2.set_xlabel('Hours from Now')
                ax2.set_ylabel('Density (cm⁻³)')
                ax2.legend()
                ax2.grid(True)
            
            # Plot 3: Magnetic Field
            ax3 = self.axes[1, 0]
            current_mag = features['magnetic_field_magnitude']
            if 'magnetic_field_magnitude' in future_pred:
                future_mag = future_pred['magnetic_field_magnitude']
                
                ax3.plot([0], [current_mag], 'ro', markersize=8, label='Current')
                ax3.plot(future_hours, future_mag, 'm-', label='24h Prediction')
                ax3.axhline(y=20, color='r', linestyle='--', label='CME Threshold')
                ax3.set_title('Magnetic Field Magnitude')
                ax3.set_xlabel('Hours from Now')
                ax3.set_ylabel('Field (nT)')
                ax3.legend()
                ax3.grid(True)
            
            # Plot 4: Dynamic Pressure
            ax4 = self.axes[1, 1]
            current_pressure = features['dynamic_pressure']
            if 'dynamic_pressure' in future_pred:
                future_pressure = future_pred['dynamic_pressure']
                
                ax4.plot([0], [current_pressure], 'ro', markersize=8, label='Current')
                ax4.plot(future_hours, future_pressure, 'c-', label='24h Prediction')
                ax4.axhline(y=5, color='r', linestyle='--', label='CME Threshold')
                ax4.set_title('Dynamic Pressure')
                ax4.set_xlabel('Hours from Now')
                ax4.set_ylabel('Pressure (nPa)')
                ax4.legend()
                ax4.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.update_results(f"Plotting error: {str(e)}")
        
    def load_data(self, source):
        """Load data from specified source"""
        try:
            # Define filetypes based on source
            if source == 'magnetometer':
                filetypes = [("CDF files", "*.cdf"), ("NetCDF files", "*.nc"), ("CSV files", "*.csv"), ("All files", "*.*")]
            elif source == 'soleriox':
                filetypes = [("CDF files", "*.cdf"), ("GTI files", "*.gti"), ("CSV files", "*.csv"), ("All files", "*.*")]
            else:
                filetypes = [("CDF files", "*.cdf"), ("CSV files", "*.csv"), ("All files", "*.*")]
            
            file_path = filedialog.askopenfilename(
                title=f"Select {source.upper()} data file",
                filetypes=filetypes
            )
            
            if not file_path:
                return
                
            # Parse data based on source and file type
            if source == 'swis':
                if file_path.endswith('.cdf'):
                    self.data_sources[source] = swis_parser.parse_cdf(file_path)
                else:
                    self.data_sources[source] = swis_parser.parse_csv(file_path)
            elif source == 'soleriox':
                if file_path.endswith('.cdf'):
                    self.data_sources[source] = soleriox_parser.parse_cdf(file_path)
                elif file_path.endswith('.gti'):
                    self.data_sources[source] = soleriox_parser.parse_gti(file_path)
                else:
                    self.data_sources[source] = soleriox_parser.parse_csv(file_path)
            elif source == 'magnetometer':
                if file_path.endswith('.cdf'):
                    self.data_sources[source] = magnetometer_parser.parse_cdf(file_path)
                elif file_path.endswith('.nc'):
                    self.data_sources[source] = magnetometer_parser.parse_netcdf(file_path)
                else:
                    self.data_sources[source] = magnetometer_parser.parse_csv(file_path)
            
            # Update status
            self.status_labels[source].config(text="Loaded", foreground="green")
            self.update_results(f"{source.upper()} data loaded successfully from {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {source} data: {str(e)}")
            self.update_results(f"Error loading {source} data: {str(e)}")
    
    def fuse_and_predict(self):
        """Fuse data and run C prediction engine"""
        try:
            # Check if at least one data source is loaded
            loaded_sources = [k for k, v in self.data_sources.items() if v is not None]
            if not loaded_sources:
                messagebox.showwarning("Warning", "Please load at least one data source first")
                return
            
            self.update_results("Starting data fusion and prediction...")
            
            # Fuse available data
            self.fused_data = self.fuse_data()
            
            # Write fused data to JSON file for C program
            with open('fused/fused_input.json', 'w') as f:
                json.dump(self.fused_data, f, indent=2)
            
            # Compile C program if needed
            self.compile_c_program()
            
            # Run C prediction engine
            result = subprocess.run(['./c_core/titanus_predictor'], 
                                  capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                raise Exception(f"C program failed: {result.stderr}")
            
            # Read prediction results
            with open('fused/prediction_output.json', 'r') as f:
                self.prediction_results = json.load(f)
            
            # Display results
            self.display_prediction_results()
            
            # Check for CME detection and send alert if needed
            if self.prediction_results.get('cme_detected', False):
                self.send_cme_alert()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_results(f"Prediction error: {str(e)}")
    
    def fuse_data(self):
        """Fuse data from all available sources"""
        fused = {
            'timestamp': datetime.now().isoformat(),
            'sources': [],
            'features': {}
        }
        
        # Process each loaded data source
        for source, data in self.data_sources.items():
            if data is not None:
                fused['sources'].append(source)
                
                # Extract features based on source type
                if source == 'swis':
                    speed_data = data.get('speed', [400])
                    density_data = data.get('density', [5])
                    temp_data = data.get('temperature', [100000])
                    
                    # Handle both single values and lists
                    fused['features']['solar_wind_speed'] = float(np.mean(speed_data) if isinstance(speed_data, list) else speed_data)
                    fused['features']['proton_density'] = float(np.mean(density_data) if isinstance(density_data, list) else density_data)
                    fused['features']['temperature'] = float(np.mean(temp_data) if isinstance(temp_data, list) else temp_data)
                    
                elif source == 'soleriox':
                    ion_flux_data = data.get('ion_flux', [1e5])
                    electron_flux_data = data.get('electron_flux', [1e6])
                    
                    # Handle both single values and lists
                    fused['features']['ion_flux'] = float(np.mean(ion_flux_data) if isinstance(ion_flux_data, list) else ion_flux_data)
                    fused['features']['electron_flux'] = float(np.mean(electron_flux_data) if isinstance(electron_flux_data, list) else electron_flux_data)
                    
                elif source == 'magnetometer':
                    mag_data = data.get('magnetic_field', [[10, 5, -8]])
                    mag_magnitude = data.get('magnetic_field_magnitude', [12.2])
                    
                    # Handle nested lists for magnetic field
                    if isinstance(mag_data, list) and len(mag_data) > 0:
                        if isinstance(mag_data[0], list):
                            # It's a list of vectors, take the mean of each component
                            mag_array = np.array(mag_data)
                            if mag_array.shape[1] >= 3:
                                fused['features']['magnetic_field_x'] = float(np.mean(mag_array[:, 0]))
                                fused['features']['magnetic_field_y'] = float(np.mean(mag_array[:, 1]))
                                fused['features']['magnetic_field_z'] = float(np.mean(mag_array[:, 2]))
                                fused['features']['magnetic_field_magnitude'] = float(np.mean(np.sqrt(
                                    mag_array[:, 0]**2 + mag_array[:, 1]**2 + mag_array[:, 2]**2)))
                        else:
                            # It's a single vector
                            if len(mag_data) >= 3:
                                fused['features']['magnetic_field_x'] = float(mag_data[0])
                                fused['features']['magnetic_field_y'] = float(mag_data[1])
                                fused['features']['magnetic_field_z'] = float(mag_data[2])
                                fused['features']['magnetic_field_magnitude'] = float(np.sqrt(
                                    mag_data[0]**2 + mag_data[1]**2 + mag_data[2]**2))
                    
                    # Use provided magnitude if available
                    if mag_magnitude and isinstance(mag_magnitude, list):
                        fused['features']['magnetic_field_magnitude'] = float(np.mean(mag_magnitude))
        
        # Add derived features
        if 'solar_wind_speed' in fused['features']:
            # Calculate dynamic pressure
            density = fused['features'].get('proton_density', 5)
            speed = fused['features']['solar_wind_speed']
            fused['features']['dynamic_pressure'] = 1.67e-6 * density * speed**2
        
        # Generate future predictions using statistical methods
        fused['predictions'] = predictor.generate_predictions(self.data_sources)
        
        return fused
    
    def compile_c_program(self):
        """Compile the C prediction engine"""
        try:
            result = subprocess.run(['make', '-C', 'c_core'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Compilation failed: {result.stderr}")
            self.update_results("C program compiled successfully")
        except Exception as e:
            raise Exception(f"Failed to compile C program: {str(e)}")
    
    def display_prediction_results(self):
        """Display prediction results in the GUI"""
        if not self.prediction_results:
            return
        
        results_text = f"Prediction Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        results_text += "="*60 + "\n\n"
        
        # CME Detection
        cme_detected = self.prediction_results.get('cme_detected', False)
        results_text += f"CME Detected: {'YES' if cme_detected else 'NO'}\n"
        results_text += f"Confidence: {self.prediction_results.get('confidence', 0):.2f}\n\n"
        
        # Feature Analysis
        results_text += "Feature Analysis:\n"
        features = self.prediction_results.get('features', {})
        for feature, value in features.items():
            results_text += f"  {feature}: {value:.3f}\n"
        
        # Future Predictions
        if 'future_predictions' in self.prediction_results:
            results_text += "\nFuture Predictions (24h):\n"
            predictions = self.prediction_results['future_predictions']
            for param, values in predictions.items():
                if values:
                    results_text += f"  {param}: {values[-1]:.3f} (trend: {self.get_trend(values)})\n"
        
        # Thresholds
        if 'thresholds_used' in self.prediction_results:
            results_text += "\nThresholds Used:\n"
            thresholds = self.prediction_results['thresholds_used']
            for threshold, value in thresholds.items():
                results_text += f"  {threshold}: {value}\n"
        
        self.update_results(results_text)
    
    def get_trend(self, values):
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        slope = (values[-1] - values[0]) / len(values)
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def send_cme_alert(self):
        """Send CME detection alert via email"""
        try:
            subject = "CME DETECTION ALERT - TITANUS System"
            body = f"""
CME Detection Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ALERT: Coronal Mass Ejection detected by TITANUS prediction system.

Prediction Details:
- Confidence: {self.prediction_results.get('confidence', 0):.2f}
- Detection Time: {datetime.now().isoformat()}
- Data Sources: {', '.join(self.fused_data.get('sources', []))}

Key Parameters:
"""
            features = self.prediction_results.get('features', {})
            for feature, value in features.items():
                body += f"- {feature}: {value:.3f}\n"
            
            body += "\nThis is an automated alert from the TITANUS CME prediction system."
            
            email_alert.send_alert(subject, body)
            self.update_results("CME alert email sent successfully")
            
        except Exception as e:
            self.update_results(f"Failed to send CME alert: {str(e)}")
    
    def train_thresholds(self):
        """Train detection thresholds using CACTUS data"""
        try:
            self.update_results("Starting threshold training...")
            
            # Run threshold training
            new_thresholds = threshold_trainer.train_thresholds()
            
            # Update model weights file
            threshold_trainer.update_c_thresholds(new_thresholds)
            
            self.update_results("Threshold training completed successfully")
            self.update_results(f"New thresholds: {new_thresholds}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Threshold training failed: {str(e)}")
            self.update_results(f"Threshold training error: {str(e)}")
    
    def log_to_database(self):
        """Log current results to database"""
        try:
            if not self.prediction_results:
                messagebox.showwarning("Warning", "No prediction results to log")
                return
            
            if not self.db_conn:
                messagebox.showerror("Error", "Database connection not available")
                return
            
            # Log to database
            db_logger.log_prediction(self.db_conn, self.prediction_results, self.fused_data)
            
            self.update_results("Results logged to database successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Database logging failed: {str(e)}")
            self.update_results(f"Database logging error: {str(e)}")
    
    def generate_pdf_report(self):
        """Generate PDF report of current results"""
        try:
            if not self.prediction_results:
                messagebox.showwarning("Warning", "No prediction results to report")
                return
            
            # Generate PDF report
            filename = report_generator.generate_report(
                self.prediction_results, 
                self.fused_data,
                self.data_sources
            )
            
            self.update_results(f"PDF report generated: {filename}")
            messagebox.showinfo("Success", f"PDF report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"PDF generation failed: {str(e)}")
            self.update_results(f"PDF generation error: {str(e)}")
    
    def plot_live_data(self):
        """Plot live data with predictions"""
        try:
            # Clear previous plots
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot data from each source
            if self.data_sources['swis']:
                self.plot_swis_data()
            if self.data_sources['soleriox']:
                self.plot_soleriox_data()
            if self.data_sources['magnetometer']:
                self.plot_magnetometer_data()
            
            # Plot predictions if available
            if self.prediction_results and 'future_predictions' in self.prediction_results:
                self.plot_predictions()
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.update_results(f"Plotting error: {str(e)}")
    
    def plot_swis_data(self):
        """Plot SWIS solar wind data"""
        data = self.data_sources['swis']
        ax = self.axes[0, 0]
        
        # Get speed data and handle different formats
        speed_data = data.get('speed', [400])
        if not isinstance(speed_data, list):
            speed_data = [speed_data]
        
        # Create time series
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             periods=len(speed_data), freq='H')
        
        ax.plot(times, speed_data, label='Solar Wind Speed', color='blue')
        ax.set_title('SWIS: Solar Wind Speed')
        ax.set_ylabel('Speed (km/s)')
        ax.legend()
        ax.grid(True)
    
    def plot_soleriox_data(self):
        """Plot SOLERIOX particle data"""
        data = self.data_sources['soleriox']
        ax = self.axes[0, 1]
        
        # Get ion flux data and handle different formats
        ion_flux_data = data.get('ion_flux', [1e5])
        if not isinstance(ion_flux_data, list):
            ion_flux_data = [ion_flux_data]
        
        # Ensure positive values for log plot
        ion_flux_data = [max(val, 1e3) for val in ion_flux_data]
        
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             periods=len(ion_flux_data), freq='H')
        
        ax.semilogy(times, ion_flux_data, label='Ion Flux', color='red')
        ax.set_title('SOLERIOX: Particle Flux')
        ax.set_ylabel('Flux (particles/cm²/s)')
        ax.legend()
        ax.grid(True)
    
    def plot_magnetometer_data(self):
        """Plot magnetometer data"""
        data = self.data_sources['magnetometer']
        ax = self.axes[1, 0]
        
        mag_field = data.get('magnetic_field', [[10, 5, -8]])
        mag_magnitude = data.get('magnetic_field_magnitude', [12.2])
        
        # Handle different data structures
        if isinstance(mag_field, list) and len(mag_field) > 0:
            # If we have pre-calculated magnitude, use it
            if mag_magnitude and isinstance(mag_magnitude, list):
                magnitude = mag_magnitude
            else:
                # Calculate magnitude from components
                if isinstance(mag_field[0], list):
                    # List of vectors
                    magnitude = [np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2) for vec in mag_field if len(vec) >= 3]
                else:
                    # Single vector
                    if len(mag_field) >= 3:
                        magnitude = [np.sqrt(mag_field[0]**2 + mag_field[1]**2 + mag_field[2]**2)]
                    else:
                        magnitude = [10.0]  # Default value
            
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 periods=len(magnitude), freq='H')
            
            ax.plot(times, magnitude, label='|B| Magnitude', color='green')
        
        ax.set_title('Magnetometer: Magnetic Field')
        ax.set_ylabel('Field Strength (nT)')
        ax.legend()
        ax.grid(True)
    
    def plot_predictions(self):
        """Plot future predictions"""
        ax = self.axes[1, 1]
        predictions = self.prediction_results['future_predictions']
        
        # Create future time series
        future_times = pd.date_range(start=datetime.now(), periods=24, freq='H')
        
        for param, values in predictions.items():
            if values and len(values) == 24:
                ax.plot(future_times, values, label=f'Predicted {param}', linestyle='--')
        
        ax.set_title('24-Hour Predictions')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        ax.grid(True)
    
    def update_results(self, text):
        """Update results text widget"""
        self.results_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {text}\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

def main():
    """Main application entry point"""
    # Create necessary directories
    os.makedirs('fused', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Initialize and run GUI
    root = tk.Tk()
    app = TitanusGUI(root)
    
    # Set up proper closing
    def on_closing():
        if app.db_conn:
            app.db_conn.close()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
