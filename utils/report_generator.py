"""
PDF Report Generator for TITANUS CME Detection System
Creates comprehensive reports with predictions, data analysis, and visualizations
"""

import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import tempfile
import json

class TitanusReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles()
        
    def custom_styles(self):
        """Define custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkred
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ))

def generate_report(prediction_results, fused_data, data_sources, output_dir='reports'):
    """Generate comprehensive PDF report"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'TITANUS_CME_Report_{timestamp}.pdf'
        filepath = os.path.join(output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        # Initialize report generator
        generator = TitanusReportGenerator()
        
        # Build report content
        story.extend(generator.create_title_page(prediction_results))
        story.append(PageBreak())
        
        story.extend(generator.create_executive_summary(prediction_results, fused_data))
        story.append(PageBreak())
        
        story.extend(generator.create_detection_analysis(prediction_results))
        story.append(PageBreak())
        
        story.extend(generator.create_data_summary(data_sources, fused_data))
        story.append(PageBreak())
        
        story.extend(generator.create_prediction_section(prediction_results))
        story.append(PageBreak())
        
        story.extend(generator.create_technical_details(prediction_results, fused_data))
        story.append(PageBreak())
        
        # Generate and add plots
        plot_files = generator.create_plots(prediction_results, data_sources)
        story.extend(generator.add_plots_to_report(plot_files))
        
        story.extend(generator.create_recommendations(prediction_results))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary plot files
        for plot_file in plot_files:
            try:
                os.remove(plot_file)
            except OSError:
                pass
        
        print(f"Report generated: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to generate report: {e}")
        raise

def create_title_page(self, prediction_results):
    """Create title page"""
    story = []
    
    # Title
    title = Paragraph("TITANUS CME Detection & Prediction Report", self.styles['CustomTitle'])
    story.append(title)
    story.append(Spacer(1, 0.5*inch))
    
    # Subtitle
    subtitle = Paragraph("Coronal Mass Ejection Analysis", self.styles['Heading2'])
    story.append(subtitle)
    story.append(Spacer(1, 0.3*inch))
    
    # Date and time
    now = datetime.now()
    date_str = now.strftime('%B %d, %Y at %H:%M UTC')
    date_para = Paragraph(f"Report Generated: {date_str}", self.styles['CustomBody'])
    story.append(date_para)
    story.append(Spacer(1, 0.5*inch))
    
    # Alert status
    cme_detected = prediction_results.get('cme_detected', False)
    confidence = prediction_results.get('confidence', 0.0)
    
    if cme_detected:
        status_text = f"<font color='red'><b>CME DETECTED</b></font><br/>Confidence Level: {confidence:.1%}"
        alert_level = "HIGH ALERT"
    else:
        status_text = "<font color='green'><b>NO CME DETECTED</b></font><br/>Quiet Solar Wind Conditions"
        alert_level = "NORMAL CONDITIONS"
    
    status_para = Paragraph(status_text, self.styles['Heading3'])
    story.append(status_para)
    story.append(Spacer(1, 0.2*inch))
    
    alert_para = Paragraph(f"Alert Level: <b>{alert_level}</b>", self.styles['CustomBody'])
    story.append(alert_para)
    
    return story

def create_executive_summary(self, prediction_results, fused_data):
    """Create executive summary"""
    story = []
    
    # Section title
    title = Paragraph("Executive Summary", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Detection summary
    cme_detected = prediction_results.get('cme_detected', False)
    confidence = prediction_results.get('confidence', 0.0)
    
    if cme_detected:
        summary_text = f"""
        The TITANUS CME detection system has identified a potential Coronal Mass Ejection (CME) event 
        with a confidence level of {confidence:.1%}. This detection is based on analysis of multi-instrument 
        space weather data including solar wind parameters, energetic particle measurements, and magnetic 
        field observations.
        """
        
        # Add space weather impact assessment
        impact_text = """
        <b>Potential Space Weather Impacts:</b><br/>
        • Geomagnetic storm activity possible<br/>
        • Satellite operations may be affected<br/>
        • Radio communications disruption likely<br/>
        • Enhanced aurora activity expected<br/>
        • Power grid fluctuations possible
        """
    else:
        summary_text = f"""
        Current analysis indicates normal solar wind conditions with no significant CME signatures detected. 
        The confidence in this assessment is {confidence:.1%}. All monitored parameters are within 
        typical ranges for quiet solar wind.
        """
        
        impact_text = """
        <b>Current Space Weather Status:</b><br/>
        • Geomagnetic conditions: Quiet<br/>
        • Satellite operations: Normal<br/>
        • Radio communications: Unaffected<br/>
        • Aurora activity: Background levels<br/>
        • Power grid: Stable
        """
    
    summary_para = Paragraph(summary_text, self.styles['CustomBody'])
    story.append(summary_para)
    story.append(Spacer(1, 0.2*inch))
    
    impact_para = Paragraph(impact_text, self.styles['CustomBody'])
    story.append(impact_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Data sources
    sources = fused_data.get('sources', []) if fused_data else []
    sources_text = f"<b>Data Sources:</b> {', '.join(sources).upper()}" if sources else "<b>Data Sources:</b> None"
    sources_para = Paragraph(sources_text, self.styles['CustomBody'])
    story.append(sources_para)
    
    return story

def create_detection_analysis(self, prediction_results):
    """Create detection analysis section"""
    story = []
    
    title = Paragraph("Detection Analysis", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Detection method
    method = prediction_results.get('prediction_method', 'Unknown')
    method_text = f"<b>Detection Method:</b> {method}"
    method_para = Paragraph(method_text, self.styles['CustomBody'])
    story.append(method_para)
    story.append(Spacer(1, 0.1*inch))
    
    # Threshold analysis
    thresholds = prediction_results.get('thresholds_used', {})
    features = prediction_results.get('features', {})
    
    if thresholds and features:
        # Create threshold comparison table
        table_data = [['Parameter', 'Measured Value', 'Threshold', 'Status']]
        
        threshold_mapping = {
            'SOLAR_WIND_THRESHOLD': ('solar_wind_speed', 'km/s'),
            'PROTON_DENSITY_THRESHOLD': ('proton_density', 'cm⁻³'),
            'DYNAMIC_PRESSURE_THRESHOLD': ('dynamic_pressure', 'nPa'),
            'MAGNETIC_FIELD_THRESHOLD': ('magnetic_field_magnitude', 'nT')
        }
        
        for threshold_key, (feature_key, unit) in threshold_mapping.items():
            if threshold_key in thresholds and feature_key in features:
                threshold_val = thresholds[threshold_key]
                measured_val = features[feature_key]
                
                if feature_key == 'temperature':
                    status = 'TRIGGERED' if measured_val < threshold_val else 'Normal'
                else:
                    status = 'TRIGGERED' if measured_val > threshold_val else 'Normal'
                
                table_data.append([
                    feature_key.replace('_', ' ').title(),
                    f"{measured_val:.2f} {unit}",
                    f"{threshold_val:.2f} {unit}",
                    status
                ])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
    
    return story

def create_data_summary(self, data_sources, fused_data):
    """Create data summary section"""
    story = []
    
    title = Paragraph("Data Summary", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Data availability
    if data_sources:
        availability_text = "<b>Instrument Data Availability:</b><br/>"
        for instrument, data in data_sources.items():
            status = "Available" if data is not None else "Not Available"
            availability_text += f"• {instrument.upper()}: {status}<br/>"
        
        availability_para = Paragraph(availability_text, self.styles['CustomBody'])
        story.append(availability_para)
        story.append(Spacer(1, 0.2*inch))
    
    # Feature summary
    if fused_data and fused_data.get('features'):
        features = fused_data['features']
        
        feature_data = [['Parameter', 'Value', 'Unit', 'Typical Range']]
        
        feature_info = {
            'solar_wind_speed': ('km/s', '300-500'),
            'proton_density': ('cm⁻³', '1-10'),
            'temperature': ('K', '50,000-200,000'),
            'dynamic_pressure': ('nPa', '1-5'),
            'magnetic_field_magnitude': ('nT', '3-10'),
            'ion_flux': ('particles/cm²/s', '10⁴-10⁶'),
            'electron_flux': ('particles/cm²/s', '10⁵-10⁷')
        }
        
        for param, value in features.items():
            if param in feature_info:
                unit, typical_range = feature_info[param]
                if isinstance(value, float):
                    if value > 1000:
                        value_str = f"{value:.2e}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                feature_data.append([
                    param.replace('_', ' ').title(),
                    value_str,
                    unit,
                    typical_range
                ])
        
        feature_table = Table(feature_data)
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(feature_table)
    
    return story

def create_prediction_section(self, prediction_results):
    """Create prediction section"""
    story = []
    
    title = Paragraph("Future Predictions", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    future_predictions = prediction_results.get('future_predictions', {})
    
    if future_predictions:
        pred_text = """
        The following 24-hour predictions are based on current solar wind conditions and 
        statistical analysis of historical data. These forecasts help assess the evolution 
        of space weather conditions.
        """
        pred_para = Paragraph(pred_text, self.styles['CustomBody'])
        story.append(pred_para)
        story.append(Spacer(1, 0.2*inch))
        
        # Create prediction summary table
        pred_data = [['Parameter', '6-Hour Forecast', '12-Hour Forecast', '24-Hour Forecast']]
        
        for param, values in future_predictions.items():
            if values and len(values) >= 24:
                param_name = param.replace('_', ' ').title()
                val_6h = values[5] if len(values) > 5 else values[-1]
                val_12h = values[11] if len(values) > 11 else values[-1]
                val_24h = values[23] if len(values) > 23 else values[-1]
                
                pred_data.append([
                    param_name,
                    f"{val_6h:.2f}",
                    f"{val_12h:.2f}",
                    f"{val_24h:.2f}"
                ])
        
        pred_table = Table(pred_data)
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(pred_table)
    else:
        no_pred_text = "No future predictions available for this analysis."
        no_pred_para = Paragraph(no_pred_text, self.styles['CustomBody'])
        story.append(no_pred_para)
    
    return story

def create_technical_details(self, prediction_results, fused_data):
    """Create technical details section"""
    story = []
    
    title = Paragraph("Technical Details", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Algorithm details
    algo_text = """
    <b>Detection Algorithm:</b><br/>
    The TITANUS system employs a multi-parameter threshold-based detection algorithm combined with 
    machine learning techniques. The detection process includes:<br/>
    • Multi-instrument data fusion<br/>
    • Feature extraction and normalization<br/>
    • Threshold-based screening<br/>
    • Confidence estimation<br/>
    • Future state prediction
    """
    algo_para = Paragraph(algo_text, self.styles['CustomBody'])
    story.append(algo_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Data processing
    processing_text = """
    <b>Data Processing:</b><br/>
    Raw instrument data is processed through the following pipeline:<br/>
    1. Data validation and quality control<br/>
    2. Coordinate system transformations<br/>
    3. Statistical analysis and feature extraction<br/>
    4. Multi-instrument data fusion<br/>
    5. Threshold application and detection logic
    """
    processing_para = Paragraph(processing_text, self.styles['CustomBody'])
    story.append(processing_para)
    story.append(Spacer(1, 0.2*inch))
    
    # System metadata
    metadata_text = f"""
    <b>System Information:</b><br/>
    • Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
    • Software Version: TITANUS v1.0<br/>
    • Data Sources: {len(fused_data.get('sources', []))} instruments<br/>
    • Processing Time: < 1 second<br/>
    • Confidence Method: Statistical threshold analysis
    """
    metadata_para = Paragraph(metadata_text, self.styles['CustomBody'])
    story.append(metadata_para)
    
    return story

def create_plots(self, prediction_results, data_sources):
    """Create visualization plots"""
    plot_files = []
    
    try:
        # Create temporary directory for plots
        temp_dir = tempfile.mkdtemp()
        
        # Plot 1: Time series data
        if data_sources:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Multi-Instrument Space Weather Data', fontsize=14, fontweight='bold')
            
            # Solar wind speed
            if data_sources.get('swis'):
                swis_data = data_sources['swis']
                times = range(len(swis_data.get('speed', [])))
                axes[0, 0].plot(times, swis_data.get('speed', []), 'b-', linewidth=2)
                axes[0, 0].set_title('Solar Wind Speed')
                axes[0, 0].set_ylabel('Speed (km/s)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Magnetic field
            if data_sources.get('magnetometer'):
                mag_data = data_sources['magnetometer']
                times = range(len(mag_data.get('magnetic_field_magnitude', [])))
                axes[0, 1].plot(times, mag_data.get('magnetic_field_magnitude', []), 'g-', linewidth=2)
                axes[0, 1].set_title('Magnetic Field Magnitude')
                axes[0, 1].set_ylabel('|B| (nT)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Particle flux
            if data_sources.get('soleriox'):
                particle_data = data_sources['soleriox']
                times = range(len(particle_data.get('ion_flux', [])))
                axes[1, 0].semilogy(times, particle_data.get('ion_flux', []), 'r-', linewidth=2)
                axes[1, 0].set_title('Ion Flux')
                axes[1, 0].set_ylabel('Flux (particles/cm²/s)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Future predictions
            future_predictions = prediction_results.get('future_predictions', {})
            if future_predictions.get('solar_wind_speed'):
                pred_times = range(24)
                axes[1, 1].plot(pred_times, future_predictions['solar_wind_speed'], 'b--', linewidth=2, label='Speed Prediction')
                if future_predictions.get('magnetic_field_magnitude'):
                    axes[1, 1].plot(pred_times, np.array(future_predictions['magnetic_field_magnitude']) * 50, 'g--', linewidth=2, label='B-field × 50')
                axes[1, 1].set_title('24-Hour Predictions')
                axes[1, 1].set_xlabel('Hours')
                axes[1, 1].set_ylabel('Predicted Values')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot1_path = os.path.join(temp_dir, 'timeseries_plot.png')
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot1_path)
        
        # Plot 2: Detection summary
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create detection confidence visualization
        confidence = prediction_results.get('confidence', 0.0)
        cme_detected = prediction_results.get('cme_detected', False)
        
        # Gauge-style plot
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta, r, 'k-', linewidth=3)
        ax.fill_between(theta[:33], 0, r[:33], color='green', alpha=0.3, label='Low Risk')
        ax.fill_between(theta[33:66], 0, r[33:66], color='yellow', alpha=0.3, label='Moderate Risk')
        ax.fill_between(theta[66:], 0, r[66:], color='red', alpha=0.3, label='High Risk')
        
        # Add confidence indicator
        confidence_angle = confidence * np.pi
        ax.plot([confidence_angle, confidence_angle], [0, 1], 'k-', linewidth=8, alpha=0.8)
        ax.plot([confidence_angle, confidence_angle], [0, 1], 'white', linewidth=4)
        
        ax.set_ylim(0, 1.2)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_title(f'CME Detection Confidence: {confidence:.1%}\nStatus: {"DETECTED" if cme_detected else "NOT DETECTED"}', 
                    fontsize=12, fontweight='bold', pad=20)
        
        plot2_path = os.path.join(temp_dir, 'confidence_gauge.png')
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot2_path)
        
    except Exception as e:
        print(f"Failed to create plots: {e}")
    
    return plot_files

def add_plots_to_report(self, plot_files):
    """Add plots to report"""
    story = []
    
    title = Paragraph("Data Visualizations", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            try:
                # Add plot image
                img = Image(plot_file, width=7*inch, height=5*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
            except Exception as e:
                print(f"Failed to add plot {plot_file}: {e}")
    
    return story

def create_recommendations(self, prediction_results):
    """Create recommendations section"""
    story = []
    
    title = Paragraph("Recommendations", self.styles['CustomHeading'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    cme_detected = prediction_results.get('cme_detected', False)
    confidence = prediction_results.get('confidence', 0.0)
    
    if cme_detected and confidence > 0.7:
        recommendations = """
        <b>High Confidence CME Detection - Immediate Actions Recommended:</b><br/>
        • Activate space weather alert protocols<br/>
        • Notify satellite operators of potential impacts<br/>
        • Monitor geomagnetic indices for storm development<br/>
        • Prepare for possible radio communication disruptions<br/>
        • Alert power grid operators of potential fluctuations<br/>
        • Continue monitoring for CME evolution and arrival time
        """
    elif cme_detected and confidence > 0.4:
        recommendations = """
        <b>Moderate Confidence CME Detection - Precautionary Measures:</b><br/>
        • Continue enhanced monitoring of solar wind conditions<br/>
        • Prepare contingency plans for space weather impacts<br/>
        • Issue advisory notices to relevant stakeholders<br/>
        • Monitor for additional confirming signatures<br/>
        • Maintain readiness for alert escalation
        """
    else:
        recommendations = """
        <b>Normal Conditions - Routine Monitoring:</b><br/>
        • Continue standard space weather monitoring<br/>
        • Maintain data quality and instrument health checks<br/>
        • Review threshold settings and algorithm performance<br/>
        • Prepare for potential future events<br/>
        • Archive data for historical analysis
        """
    
    rec_para = Paragraph(recommendations, self.styles['CustomBody'])
    story.append(rec_para)
    story.append(Spacer(1, 0.3*inch))
    
    # Contact information
    contact_text = """
    <b>Contact Information:</b><br/>
    For questions about this report or the TITANUS system:<br/>
    • Technical Support: titanus-support@space-weather.org<br/>
    • Emergency Notifications: alerts@space-weather.org<br/>
    • System Status: https://titanus-status.space-weather.org
    """
    contact_para = Paragraph(contact_text, self.styles['CustomBody'])
    story.append(contact_para)
    
    return story

# Bind methods to class
TitanusReportGenerator.create_title_page = create_title_page
TitanusReportGenerator.create_executive_summary = create_executive_summary
TitanusReportGenerator.create_detection_analysis = create_detection_analysis
TitanusReportGenerator.create_data_summary = create_data_summary
TitanusReportGenerator.create_prediction_section = create_prediction_section
TitanusReportGenerator.create_technical_details = create_technical_details
TitanusReportGenerator.create_plots = create_plots
TitanusReportGenerator.add_plots_to_report = add_plots_to_report
TitanusReportGenerator.create_recommendations = create_recommendations

if __name__ == "__main__":
    # Test report generation
    sample_prediction = {
        'cme_detected': True,
        'confidence': 0.85,
        'prediction_method': 'C_THRESHOLD_ENGINE',
        'features': {
            'solar_wind_speed': 650,
            'proton_density': 12,
            'temperature': 75000,
            'dynamic_pressure': 8.5,
            'magnetic_field_magnitude': 18
        },
        'future_predictions': {
            'solar_wind_speed': [640] * 24,
            'magnetic_field_magnitude': [16] * 24
        },
        'thresholds_used': {
            'SOLAR_WIND_THRESHOLD': 600,
            'PROTON_DENSITY_THRESHOLD': 10
        }
    }
    
    sample_fused_data = {
        'sources': ['swis', 'magnetometer'],
        'features': sample_prediction['features']
    }
    
    sample_data_sources = {
        'swis': {'speed': [400, 500, 650, 620]},
        'magnetometer': {'magnetic_field_magnitude': [5, 8, 18, 16]},
        'soleriox': {'ion_flux': [1e5, 2e5, 1e6, 8e5]}
    }
    
    try:
        report_path = generate_report(sample_prediction, sample_fused_data, sample_data_sources)
        print(f"Test report generated successfully: {report_path}")
    except Exception as e:
        print(f"Test report generation failed: {e}")
