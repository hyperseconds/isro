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
from matplotlib.backends.backend_tkagg import FigureCanvasTkinter
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
        
        # Data Loading Section
        data_frame = ttk.LabelFrame(main_frame, text="Data Sources", padding="10")
        data_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(data_frame, text="Load SWIS Data", 
                  command=lambda: self.load_data('swis')).grid(row=0, column=0, padx=5)
        ttk.Button(data_frame, text="Load SOLERIOX Data", 
                  command=lambda: self.load_data('soleriox')).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Load Magnetometer Data", 
                  command=lambda: self.load_data('magnetometer')).grid(row=0, column=2, padx=5)
        
        # Status indicators
        self.status_labels = {}
        for i, source in enumerate(['swis', 'soleriox', 'magnetometer']):
            self.status_labels[source] = ttk.Label(data_frame, text="Not Loaded", 
                                                  foreground="red")
            self.status_labels[source].grid(row=1, column=i, padx=5)
        
        # Control Buttons Section
        control_frame = ttk.LabelFrame(main_frame, text="Operations", padding="10")
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="Fuse + Predict", 
                  command=self.fuse_and_predict).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Train Thresholds", 
                  command=self.train_thresholds).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Log to DB", 
                  command=self.log_to_database).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Generate PDF", 
                  command=self.generate_pdf_report).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Plot Live", 
                  command=self.plot_live_data).grid(row=0, column=4, padx=5)
        
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
        self.canvas = FigureCanvasTkinter(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
    def load_data(self, source):
        """Load data from specified source"""
        try:
            file_path = filedialog.askopenfilename(
                title=f"Select {source.upper()} data file",
                filetypes=[("CDF files", "*.cdf"), ("CSV files", "*.csv"), ("All files", "*.*")]
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
                else:
                    self.data_sources[source] = soleriox_parser.parse_csv(file_path)
            elif source == 'magnetometer':
                if file_path.endswith('.cdf'):
                    self.data_sources[source] = magnetometer_parser.parse_cdf(file_path)
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
                    fused['features']['solar_wind_speed'] = float(np.mean(data.get('speed', [400])))
                    fused['features']['proton_density'] = float(np.mean(data.get('density', [5])))
                    fused['features']['temperature'] = float(np.mean(data.get('temperature', [100000])))
                    
                elif source == 'soleriox':
                    fused['features']['ion_flux'] = float(np.mean(data.get('ion_flux', [1e5])))
                    fused['features']['electron_flux'] = float(np.mean(data.get('electron_flux', [1e6])))
                    
                elif source == 'magnetometer':
                    mag_data = data.get('magnetic_field', [10, 5, -8])
                    if len(mag_data) >= 3:
                        fused['features']['magnetic_field_x'] = float(mag_data[0])
                        fused['features']['magnetic_field_y'] = float(mag_data[1])
                        fused['features']['magnetic_field_z'] = float(mag_data[2])
                        fused['features']['magnetic_field_magnitude'] = float(np.sqrt(
                            mag_data[0]**2 + mag_data[1]**2 + mag_data[2]**2))
        
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
        
        # Create time series
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             periods=len(data.get('speed', [400])), freq='H')
        
        ax.plot(times, data.get('speed', [400]), label='Solar Wind Speed', color='blue')
        ax.set_title('SWIS: Solar Wind Speed')
        ax.set_ylabel('Speed (km/s)')
        ax.legend()
        ax.grid(True)
    
    def plot_soleriox_data(self):
        """Plot SOLERIOX particle data"""
        data = self.data_sources['soleriox']
        ax = self.axes[0, 1]
        
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             periods=len(data.get('ion_flux', [1e5])), freq='H')
        
        ax.semilogy(times, data.get('ion_flux', [1e5]), label='Ion Flux', color='red')
        ax.set_title('SOLERIOX: Particle Flux')
        ax.set_ylabel('Flux (particles/cm²/s)')
        ax.legend()
        ax.grid(True)
    
    def plot_magnetometer_data(self):
        """Plot magnetometer data"""
        data = self.data_sources['magnetometer']
        ax = self.axes[1, 0]
        
        mag_field = data.get('magnetic_field', [10, 5, -8])
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             periods=len(mag_field), freq='H')
        
        if len(mag_field) >= 3:
            magnitude = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in 
                        zip([mag_field[0]], [mag_field[1]], [mag_field[2]])]
            ax.plot(times[:len(magnitude)], magnitude, label='|B| Magnitude', color='green')
        
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
