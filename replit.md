# TITANUS - CME Detection & Prediction System

## Overview

TITANUS (Temporal Integration of Terrestrial Activity and Near-Universe Surveillance) is a comprehensive space weather monitoring system designed to detect and predict Coronal Mass Ejections (CMEs) using multi-instrument space weather data. The system combines a high-performance C detection engine with a Python GUI interface for real-time CME detection and 24-hour space weather forecasting.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Hybrid Python + C Architecture
The system uses a two-tier architecture:
- **Frontend**: Python-based GUI using Tkinter for user interaction
- **Backend**: High-performance C engine for real-time data processing and CME detection
- **Data Processing**: Python modules for data parsing, fusion, and analysis

### Technology Stack
- **Languages**: Python 3.8+, C
- **GUI Framework**: Tkinter
- **Database**: PostgreSQL (primary) with SQLite fallback
- **Data Visualization**: Matplotlib
- **Scientific Computing**: NumPy, Pandas, SciPy
- **Machine Learning**: Scikit-learn
- **Report Generation**: ReportLab, Matplotlib

## Key Components

### Data Parsers (`parsers/`)
Handles multiple space weather instrument data formats:
- **SWIS Parser**: Solar Wind Ion Spectrometer data (speed, density, temperature)
- **SOLERIOX Parser**: Energetic Particle Detector data (ion/electron flux)
- **Magnetometer Parser**: Magnetic field measurements (GSE/GSM coordinates)

Each parser supports both CDF (Common Data Format) and CSV formats with automatic fallback capabilities.

### Database Layer (`models.py`)
- **Primary**: PostgreSQL for production environments
- **Fallback**: SQLite for development/offline use
- **Tables**: Instrument data, predictions, alerts, and metadata
- **Connection Management**: Automatic failover between PostgreSQL and SQLite

### Utilities (`utils/`)
Core system utilities:
- **Database Logger**: Logs predictions and instrument data
- **Report Generator**: Creates PDF reports with visualizations
- **Email Alert System**: Automated notifications for CME events
- **Predictor**: Time-series forecasting for space weather parameters

### Training Module (`training/`)
Machine learning components for threshold optimization:
- Uses CACTUS CME catalog data for training
- Random Forest classifier for threshold optimization
- Performance metrics tracking (accuracy, precision, recall, F1-score)

## Data Flow

1. **Data Ingestion**: Multiple instrument data sources (SWIS, SOLERIOX, Magnetometer)
2. **Data Parsing**: Format-specific parsers convert raw data to standardized format
3. **Data Fusion**: Combines multi-instrument data with quality metrics
4. **CME Detection**: C engine processes fused data using adaptive thresholds
5. **Prediction**: Generates 24-hour forecasts using time-series analysis
6. **Alert Generation**: Sends email notifications for detected CME events
7. **Report Generation**: Creates comprehensive PDF reports
8. **Database Storage**: Logs all results for historical analysis

### Data Processing Pipeline
- Input: Multi-instrument space weather data
- Processing: Data fusion with quality assessment
- Analysis: Threshold-based detection with ML enhancement
- Output: CME detection results and future predictions
- Storage: PostgreSQL database with structured logging

## External Dependencies

### Required Python Packages
- **Core**: numpy, pandas, matplotlib, psycopg2, scikit-learn, scipy
- **GUI**: tkinter (standard library)
- **Reports**: reportlab
- **Data Formats**: cdflib (for CDF files)
- **Database**: psycopg2 (PostgreSQL), sqlite3 (fallback)

### Environment Variables
- **Database**: PGHOST, PGDATABASE, PGUSER, PGPASSWORD, PGPORT
- **Email**: SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, USE_TLS
- **Alerts**: ALERT_RECIPIENTS

### External Data Sources
- CACTUS CME catalog for training data
- Real-time space weather instrument feeds
- Historical space weather databases

## Deployment Strategy

### Development Environment
- SQLite database for local development
- Local file-based data sources
- Environment variables for configuration

### Production Environment
- PostgreSQL database for data persistence
- Email server integration for alerts
- Automated data ingestion from space weather APIs
- Scheduled threshold training and optimization

### Configuration Management
- Environment variables for sensitive data
- JSON configuration files for application settings
- Fallback mechanisms for missing configurations

### Monitoring and Alerts
- Database logging for all system activities
- Email alerts for CME detection events
- PDF reports for operational summaries
- Performance metrics tracking for model optimization

The system is designed to be resilient with automatic fallbacks (PostgreSQL → SQLite, CDF → CSV) and supports both real-time operation and batch processing modes.