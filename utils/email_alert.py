"""
Email Alert System for TITANUS CME Detection
Sends automated alerts when CME events are detected
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import json

class EmailAlertSystem:
    def __init__(self):
        # Email configuration from environment variables
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('SENDER_EMAIL', 'titanus-alerts@space-weather.org')
        self.sender_password = os.getenv('SENDER_PASSWORD', 'default_password')
        self.use_tls = os.getenv('USE_TLS', 'True').lower() == 'true'
        
        # Default recipient list
        self.default_recipients = self.load_recipients()
        
    def load_recipients(self):
        """Load recipient list from environment or config file"""
        # Try environment variable first
        recipients_env = os.getenv('ALERT_RECIPIENTS', '')
        if recipients_env:
            return [email.strip() for email in recipients_env.split(',')]
        
        # Try config file
        try:
            with open('config/alert_recipients.json', 'r') as f:
                config = json.load(f)
                return config.get('recipients', [])
        except FileNotFoundError:
            # Default recipients for testing
            return ['admin@space-weather.org', 'operations@space-weather.org']
    
    def send_alert(self, subject, body, recipients=None, attachment_path=None, priority='normal'):
        """Send email alert"""
        try:
            if recipients is None:
                recipients = self.default_recipients
            
            if not recipients:
                print("No recipients configured for email alerts")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Set priority
            if priority.lower() == 'high':
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            elif priority.lower() == 'low':
                msg['X-Priority'] = '5'
                msg['X-MSMail-Priority'] = 'Low'
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                self.attach_file(msg, attachment_path)
            
            # Connect to server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"Alert email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
            return False
    
    def attach_file(self, msg, file_path):
        """Attach file to email message"""
        try:
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(file_path)}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            print(f"Failed to attach file {file_path}: {e}")

def send_cme_detection_alert(prediction_results, fused_data=None):
    """Send CME detection alert"""
    alert_system = EmailAlertSystem()
    
    confidence = prediction_results.get('confidence', 0.0)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Determine alert priority
    if confidence > 0.8:
        priority = 'high'
        urgency = 'HIGH PRIORITY'
    elif confidence > 0.5:
        priority = 'normal'
        urgency = 'MODERATE PRIORITY'
    else:
        priority = 'low'
        urgency = 'LOW PRIORITY'
    
    subject = f"[TITANUS] CME DETECTION ALERT - {urgency} - {timestamp}"
    
    # Create alert body
    body = create_cme_alert_body(prediction_results, fused_data, urgency, timestamp)
    
    return alert_system.send_alert(subject, body, priority=priority)

def send_system_alert(alert_type, message, severity='INFO'):
    """Send system status alert"""
    alert_system = EmailAlertSystem()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    if severity.upper() == 'ERROR':
        priority = 'high'
        subject_prefix = '[TITANUS ERROR]'
    elif severity.upper() == 'WARNING':
        priority = 'normal'
        subject_prefix = '[TITANUS WARNING]'
    else:
        priority = 'low'
        subject_prefix = '[TITANUS INFO]'
    
    subject = f"{subject_prefix} {alert_type} - {timestamp}"
    
    body = f"""
TITANUS System Alert
{'=' * 50}

Alert Type: {alert_type}
Severity: {severity}
Timestamp: {timestamp}

Message:
{message}

System Information:
- Alert generated automatically by TITANUS monitoring system
- For technical support, contact: titanus-support@space-weather.org
- System status: https://titanus-status.space-weather.org

This is an automated message from the TITANUS CME Detection System.
"""
    
    return alert_system.send_alert(subject, body, priority=priority)

def create_cme_alert_body(prediction_results, fused_data, urgency, timestamp):
    """Create CME alert email body"""
    confidence = prediction_results.get('confidence', 0.0)
    features = prediction_results.get('features', {})
    thresholds = prediction_results.get('thresholds_used', {})
    sources = fused_data.get('sources', []) if fused_data else []
    
    body = f"""
CORONAL MASS EJECTION DETECTION ALERT
{'=' * 60}

ALERT LEVEL: {urgency}
DETECTION TIME: {timestamp}
CONFIDENCE LEVEL: {confidence:.1%}

SUMMARY:
The TITANUS CME detection system has identified a potential Coronal Mass Ejection 
event based on multi-instrument space weather data analysis. Immediate attention 
and appropriate response measures are recommended.

DETECTION DETAILS:
{'─' * 40}
• Detection Method: {prediction_results.get('prediction_method', 'Unknown')}
• Confidence Score: {confidence:.3f} ({confidence:.1%})
• Data Sources: {', '.join(sources).upper() if sources else 'Multiple instruments'}
• Analysis Timestamp: {timestamp}
"""
    
    # Add space weather parameters
    if features:
        body += f"""
SPACE WEATHER PARAMETERS:
{'─' * 40}
"""
        
        param_info = {
            'solar_wind_speed': ('Solar Wind Speed', 'km/s', 'Enhanced speed indicates CME passage'),
            'proton_density': ('Proton Density', 'cm⁻³', 'Density enhancement typical of CME'),
            'temperature': ('Proton Temperature', 'K', 'Temperature depression indicates CME'),
            'dynamic_pressure': ('Dynamic Pressure', 'nPa', 'Pressure enhancement from CME'),
            'magnetic_field_magnitude': ('Magnetic Field', 'nT', 'Field enhancement in CME'),
            'ion_flux': ('Ion Flux', 'particles/cm²/s', 'Energetic particle enhancement'),
            'electron_flux': ('Electron Flux', 'particles/cm²/s', 'Electron enhancement')
        }
        
        for param, value in features.items():
            if param in param_info:
                name, unit, description = param_info[param]
                if isinstance(value, float):
                    if value > 1000:
                        value_str = f"{value:.2e}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                body += f"• {name}: {value_str} {unit}\n"
                
                # Add threshold comparison if available
                threshold_key = f"{param.upper()}_THRESHOLD"
                if threshold_key in thresholds:
                    threshold_val = thresholds[threshold_key]
                    if param == 'temperature':
                        status = 'BELOW THRESHOLD' if value < threshold_val else 'Above threshold'
                    else:
                        status = 'ABOVE THRESHOLD' if value > threshold_val else 'Below threshold'
                    body += f"  └─ Threshold: {threshold_val} {unit} ({status})\n"
                
                body += f"  └─ Significance: {description}\n\n"
    
    # Add future predictions if available
    future_predictions = prediction_results.get('future_predictions', {})
    if future_predictions:
        body += f"""
24-HOUR FORECAST:
{'─' * 40}
"""
        for param, values in future_predictions.items():
            if values and len(values) >= 12:
                param_name = param.replace('_', ' ').title()
                current_val = values[0]
                forecast_12h = values[11] if len(values) > 11 else values[-1]
                
                trend = "increasing" if forecast_12h > current_val else "decreasing" if forecast_12h < current_val else "stable"
                body += f"• {param_name}: {current_val:.2f} → {forecast_12h:.2f} (12h, {trend})\n"
    
    # Add impact assessment
    body += f"""
POTENTIAL IMPACTS:
{'─' * 40}
"""
    
    if confidence > 0.7:
        body += """• HIGH PROBABILITY of geomagnetic storm development
• Satellite operations may experience anomalies or require safing
• HF radio communications likely to be degraded or blacked out
• GPS accuracy may be significantly reduced
• Power grid fluctuations and possible outages in extreme cases
• Enhanced aurora activity at lower latitudes than normal
• Radiation hazard for astronauts and high-altitude flights
"""
    elif confidence > 0.4:
        body += """• MODERATE PROBABILITY of geomagnetic disturbances
• Minor satellite operations impacts possible
• HF radio communications may experience some degradation
• GPS accuracy could be reduced
• Minor power grid fluctuations possible
• Aurora activity may be enhanced at typical latitudes
"""
    else:
        body += """• LOW PROBABILITY of significant impacts
• Minimal effects on technological systems expected
• Continue monitoring for event evolution
• Normal space weather precautions sufficient
"""
    
    # Add recommended actions
    body += f"""
RECOMMENDED ACTIONS:
{'─' * 40}
"""
    
    if confidence > 0.7:
        body += """□ IMMEDIATE: Activate space weather alert protocols
□ IMMEDIATE: Notify satellite operators and flight operations centers
□ IMMEDIATE: Alert power grid operators in affected regions
□ URGENT: Issue public space weather warnings
□ URGENT: Monitor for additional CME signatures and arrival time updates
□ CONTINUOUS: Track geomagnetic indices and solar wind conditions
"""
    elif confidence > 0.4:
        body += """□ PROMPT: Prepare space weather contingency plans
□ PROMPT: Issue advisory notices to relevant stakeholders
□ ONGOING: Enhanced monitoring of solar wind conditions
□ ONGOING: Maintain readiness for alert escalation
"""
    else:
        body += """□ ROUTINE: Continue standard space weather monitoring
□ ROUTINE: Review data for additional confirming signatures
□ ROUTINE: Maintain normal operational procedures
"""
    
    body += f"""
CONTACT INFORMATION:
{'─' * 40}
• Emergency Hotline: +1-XXX-XXX-XXXX
• Technical Support: titanus-support@space-weather.org
• Operations Center: operations@space-weather.org
• System Status: https://titanus-status.space-weather.org

IMPORTANT NOTICE:
This alert is generated automatically by the TITANUS CME detection system.
For the most current information and official space weather advisories,
please consult NOAA Space Weather Prediction Center (www.swpc.noaa.gov).

Alert ID: TITANUS-{timestamp.replace(' ', 'T').replace(':', '')}-{confidence:.0%}
Generated: {timestamp}
System: TITANUS v1.0 CME Detection & Prediction
"""
    
    return body

def send_test_alert():
    """Send test alert to verify email system"""
    alert_system = EmailAlertSystem()
    
    subject = "[TITANUS TEST] Email Alert System Test"
    body = f"""
This is a test message from the TITANUS email alert system.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
System: TITANUS v1.0
Purpose: Email system verification

If you receive this message, the email alert system is functioning properly.

Contact: titanus-support@space-weather.org
"""
    
    return alert_system.send_alert(subject, body, priority='low')

def send_alert(subject, body, recipients=None, attachment_path=None):
    """Main function for sending alerts (for external use)"""
    alert_system = EmailAlertSystem()
    return alert_system.send_alert(subject, body, recipients, attachment_path)

if __name__ == "__main__":
    # Test the email alert system
    print("Testing TITANUS email alert system...")
    
    # Send test alert
    success = send_test_alert()
    if success:
        print("Test alert sent successfully!")
    else:
        print("Test alert failed!")
    
    # Test CME detection alert
    sample_prediction = {
        'cme_detected': True,
        'confidence': 0.85,
        'prediction_method': 'C_THRESHOLD_ENGINE',
        'features': {
            'solar_wind_speed': 650,
            'proton_density': 12,
            'temperature': 75000,
            'dynamic_pressure': 8.5
        },
        'thresholds_used': {
            'SOLAR_WIND_THRESHOLD': 600,
            'PROTON_DENSITY_THRESHOLD': 10
        }
    }
    
    sample_fused_data = {
        'sources': ['swis', 'magnetometer']
    }
    
    success = send_cme_detection_alert(sample_prediction, sample_fused_data)
    if success:
        print("CME detection alert sent successfully!")
    else:
        print("CME detection alert failed!")
