# TITANUS - CME Detection & Prediction System

A hybrid Python + C scientific pipeline for Coronal Mass Ejection (CME) detection and prediction using multi-instrument space weather data with advanced forecasting capabilities.

![TITANUS System](https://img.shields.io/badge/TITANUS-CME%20Detection-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![C](https://img.shields.io/badge/C-Standard-orange) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)

## 🌟 Overview

TITANUS (Temporal Integration of Terrestrial Activity and Near-Universe Surveillance) is a comprehensive space weather monitoring system designed to detect and predict Coronal Mass Ejections (CMEs) using data from multiple space-based instruments. The system combines real-time data processing with machine learning and statistical analysis to provide accurate CME detection and future state prediction.

### Key Features

- **Multi-Instrument Data Integration**: SWIS, SOLERIOX, and Magnetometer data fusion
- **Hybrid Python + C Architecture**: High-performance C engine with Python GUI
- **Real-time CME Detection**: Threshold-based and ML-enhanced detection algorithms
- **Future Prediction Capabilities**: 24-hour space weather forecasting
- **Automated Alert System**: Email notifications for CME events
- **Comprehensive Reporting**: PDF reports with visualizations and analysis
- **Database Integration**: PostgreSQL storage with SQLite fallback
- **Adaptive Thresholds**: Machine learning-based threshold optimization

## System Architecture

How TITANUS Predicts Future CME Waves
TITANUS uses a hybrid approach that combines several sophisticated methods to predict space weather events. Here's how it works:

🔬 1. Physics-Based Feature Detection
The system analyzes real space weather data to detect specific physical signatures that occur before and during CME events:

Magnetic Field Rotation: Detects when Earth's magnetic field suddenly rotates (a key CME signature)
Solar Wind Speed Changes: Monitors sudden increases in solar wind speed (600+ km/s indicates trouble)
Particle Flux Variations: Tracks energetic particles that arrive before the main CME
Pressure Changes: Measures dynamic pressure increases that push on Earth's magnetosphere
🎯 2. Threshold-Based Detection
The C engine uses carefully trained thresholds to identify CME events:

Speed Threshold: 600 km/s (normal solar wind is ~400 km/s)
Density Threshold: 10 particles/cm³ (normal is ~5)
Temperature Threshold: 50,000 K (much hotter than normal)
Magnetic Field Threshold: 20 nT (stronger than typical)
📊 3. Multi-Method Prediction
For future forecasting, the system combines 4 different mathematical approaches:

Linear Trend: Extends current trends forward
Exponential Smoothing: Gives more weight to recent data
Polynomial Fitting: Captures curved patterns in the data
Seasonal Analysis: Identifies repeating patterns
🧠 4. Machine Learning Enhancement
The system uses a Random Forest classifier to:

Learn from historical CME events (CACTUS catalog data)
Optimize detection thresholds automatically
Improve accuracy over time
Reduce false alarms
⚡ 5. Real-Time Processing
The high-speed C engine processes data in real-time:

Analyzes multiple data streams simultaneously
Calculates gradients and variance in magnetic fields
Detects sudden changes that indicate incoming CMEs
Provides confidence levels for predictions
🎯 How It Predicts the Future
Pattern Recognition: Identifies current space weather patterns
Trend Analysis: Calculates how these patterns are changing
Mathematical Extrapolation: Projects these trends 24 hours forward
Physical Constraints: Ensures predictions make physical sense
Uncertainty Calculation: Provides confidence levels for each prediction
🌊 Why This Works for CMEs
CMEs don't appear instantly - they have warning signs:

Fast particles arrive first (30 minutes to 2 hours ahead)
Magnetic field changes precede the main impact
Solar wind conditions show characteristic signatures
Multi-instrument data provides cross-confirmation
This isn't traditional AI like ChatGPT, but rather scientific AI that combines physics knowledge with advanced statistical methods to predict dangerous space weather events that could damage satellites, power grids, and communication systems.

# TITANUS: CME Detection and Prediction System

TITANUS is a hybrid AI-based system designed to detect and forecast Coronal Mass Ejections (CMEs) using satellite payload data. It combines physics-based algorithms and time-series analysis with embedded C code for real-time performance.

This section outlines the core scientific and mathematical formulas used in TITANUS, along with their purposes and where they are used in the system.

---

## 1. Linear Gradient Calculation

**Formula:**

    gradient = (n·Σ(x·y) - Σx·Σy) / (n·Σ(x²) - (Σx)²)

**Purpose:**  
Calculates how fast a quantity (like particle flux or wind speed) is increasing or decreasing over time.

**Why It’s Needed:**  
CMEs are usually preceded by rapid changes in solar particle data.

**Used In:**  
`c_core/feature_extractor.c` and `utils/predictor.py` to compute flux and speed gradients.

---

## 2. Variance

**Formula:**

    mean = (1/n) · Σ(values)
    variance = (1/(n-1)) · Σ(values - mean)²

**Purpose:**  
Measures the spread or fluctuation in a dataset.

**Why It’s Needed:**  
Unusual variance in particle data can indicate solar disturbances.

**Used In:**  
Preprocessing and feature selection logic.

---

## 3. Magnetic Field Rotation

**Formula:**

    |B| = √(Bx² + By² + Bz²)
    cos(θ) = (B · B_expected) / (|B| · |B_expected|)
    θ = arccos(cos(θ)) in degrees

**Purpose:**  
Quantifies how much the magnetic field direction changes.

**Why It’s Needed:**  
Large magnetic rotations are characteristic of CME structures.

**Used In:**  
Magnetometer (MAG) parser and predictor module.

---

## 4. Plasma Beta

**Formula:**

    β = (n · k · T) / (B² / 2μ₀)

**Purpose:**  
Compares plasma pressure to magnetic pressure.

**Why It’s Needed:**  
Helps detect energy imbalances typical during CME onset.

**Used In:**  
Advanced model training or extended physical analysis.

---

## 5. Alfvén Speed

**Formula:**

    vA = B / √(μ₀ · ρ)

**Purpose:**  
Calculates the speed at which magnetic disturbances travel.

**Why It’s Needed:**  
Important for identifying magnetohydrodynamic (MHD) wave propagation from solar events.

**Used In:**  
Derived features for threshold tuning.

---

## 6. Dynamic Pressure

**Formula:**

    P = ρ · v²

**Purpose:**  
Calculates force exerted by moving plasma.

**Why It’s Needed:**  
Spikes in dynamic pressure often occur during shock-fronts of CMEs.

**Used In:**  
Predictor logic and report generation.

---

## 7. CME Detection Logic

**Logic:**

    detection_flags = 0
    if solar_wind_speed > 600 km/s: detection_flags += 1
    if proton_density > 10 cm⁻³: detection_flags += 1
    if magnetic_rotation > 30°: detection_flags += 1
    if dynamic_pressure > 5 nPa: detection_flags += 1
    if ion_flux > 1e6 particles/cm²/s: detection_flags += 1
    if temperature < 50000 K: detection_flags += 1
    CME_detected = (detection_flags >= 3)

**Purpose:**  
Simple rule-based system to classify a potential CME event.

**Why It’s Needed:**  
Provides a fast, explainable method of detection before AI prediction.

**Used In:**  
`c_core/main.c` and `utils/predictor.py`

---

## 8. Confidence Score

**Formula:**

    confidence = Σ(threshold_excess_ratio) / number_of_triggered_conditions + 0.5

    where threshold_excess_ratio = (value / threshold - 1) × 0.2

**Purpose:**  
Calculates how confidently the system can claim CME presence.

**Why It’s Needed:**  
Gives human users a measure of certainty.

**Used In:**  
Report generation and alert modules.

---

## 9. Ensemble Prediction

**Formula:**

    prediction = w1·linear + w2·exponential + w3·polynomial + w4·seasonal

    where w1=0.3, w2=0.25, w3=0.25, w4=0.2

**Purpose:**  
Combines multiple forecasting models to improve accuracy.

**Why It’s Needed:**  
Each model captures different patterns (trend, curve, seasonality).

**Used In:**  
`utils/predictor.py` forecast pipeline.

---

## 10. Linear Regression

**Formula:**

    y = m·x + b

**Purpose:**  
Predicts future values based on a straight-line fit.

**Why It’s Needed:**  
Gives simple, fast trend estimation.

**Used In:**  
Short-term solar wind forecasting.

---

## 11. Exponential Smoothing

**Formula:**

    S[i] = α·y[i] + (1-α)·S[i-1]

**Purpose:**  
Gives more weight to recent data, good for volatile environments.

**Used In:**  
Flux and speed smoothing in prediction.

---

## 12. Polynomial Extrapolation

**Formula:**

    p(x) = aₙxⁿ + ... + a₁x + a₀

**Purpose:**  
Fits curves to non-linear data.

**Why It’s Needed:**  
Handles events with sharp bends (like CME onset or recovery).

**Used In:**  
Trend modeling.

---

## 13. Seasonal Decomposition

**Steps:**

1. Estimate period using autocorrelation
2. Extract seasonal signal
3. Detrend the data
4. Combine trend + seasonality for prediction

**Purpose:**  
Isolates daily/periodic variations in solar parameters.

**Used In:**  
Forecasting long-term cyclical behavior.

---

## 14. Spectral Power

**Formula:**

    power = (1/(n-1)) · Σ(derivative[i]²)

**Purpose:**  
Measures energy in fluctuations.

**Why It’s Needed:**  
High-frequency variations may signal instability.

---

## 15. Flux Dropout Detection

**Formula:**

    dropout_ratio = (baseline - min_value) / baseline

**Purpose:**  
Detects sudden drops in particle flux, a known CME indicator.

**Used In:**  
`predictor.py` decision logic.

---

## 16. Future Extrapolation Functions

**Formula Examples:**

    v(h) = v0 · (1 - h/24 · 0.1)
    n(h) = n0 · (1 + sin(π·h/24) · 0.2)

**Purpose:**  
Generates hourly future forecasts for up to 24 hours.

**Used In:**  
User-facing dashboards and alerts.

---

## 17. Physical Constraints

**Enforced Limits:**

- Wind speed: 200 – 2000 km/s
- Proton density: 0.1 – 100 cm⁻³
- Temperature: 1,000 – 10⁷ K
- Magnetic field: 0.1 – 100 nT

**Why It’s Needed:**  
Prevents invalid or unphysical results.

**Used In:**  
Validation during fusion and prediction.

---

## 18. Multi-Instrument Fusion

**Formula:**

    fused_value = w1·SWIS + w2·SOLEXS + w3·MAG

**Weights:**

- SWIS: 0.4  
- SOLEXS: 0.3  
- MAG: 0.3

**Purpose:**  
Combines readings from multiple instruments for robustness.

**Used In:**  
`main.py` during `fuse_payloads_from_folder()`.

---

## Summary

TITANUS relies on a strong mix of physics equations, time-series forecasting, and detection heuristics. Each formula is designed to enhance interpretability, reliability, and real-time responsiveness of the CME detection pipeline.
