/*
 * Model Weights and Thresholds for TITANUS CME Detection
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

// Detection thresholds for CME identification
#define SOLAR_WIND_THRESHOLD      600.0    // km/s - Enhanced solar wind speed
#define PROTON_DENSITY_THRESHOLD  10.0     // cm⁻³ - Enhanced proton density
#define TEMPERATURE_THRESHOLD     50000.0  // K - Temperature depression threshold
#define DYNAMIC_PRESSURE_THRESHOLD 5.0     // nPa - Enhanced dynamic pressure
#define ION_FLUX_THRESHOLD        1e6      // particles/cm²/s - Enhanced ion flux
#define ELECTRON_FLUX_THRESHOLD   1e7      // particles/cm²/s - Enhanced electron flux
#define MAGNETIC_ROTATION_THRESHOLD 30.0   // degrees - Magnetic field rotation
#define MAGNETIC_FIELD_THRESHOLD  20.0     // nT - Enhanced magnetic field magnitude

// Advanced thresholds for sophisticated detection
#define PROTON_BETA_THRESHOLD     1.0      // Plasma beta parameter
#define ALFVEN_SPEED_THRESHOLD    50.0     // km/s - Alfvén speed
#define FLUX_DROPOUT_THRESHOLD    0.3      // 30% flux dropout
#define SPECTRAL_POWER_THRESHOLD  100.0    // Spectral power enhancement

// Detection logic parameters
#define MIN_DETECTION_FLAGS       3        // Minimum number of thresholds to trigger detection
#define CONFIDENCE_SCALING        0.8      // Confidence scaling factor

// Prediction parameters
#define PREDICTION_HOURS          24       // Hours to predict into future
#define NUM_PREDICTION_PARAMS     4        // Number of parameters to predict

// Time series analysis parameters
#define MIN_DATA_POINTS           5        // Minimum data points for analysis
#define TIME_WINDOW_HOURS         6        // Time window for gradient calculation
#define BASELINE_WINDOW_HOURS     12       // Baseline window for comparison

// Physical constants (CGS units where applicable)
#define PROTON_MASS              1.67e-27  // kg
#define BOLTZMANN_CONSTANT       1.38e-23  // J/K
#define MU_0                     4e-7*3.14159 // H/m

// Mathematical constants
#ifndef M_PI
#define M_PI                     3.14159265358979323846
#endif

// Quality control thresholds
#define MAX_SOLAR_WIND_SPEED     2000.0    // km/s - Maximum realistic speed
#define MAX_PROTON_DENSITY       100.0     // cm⁻³ - Maximum realistic density
#define MAX_TEMPERATURE          1e7       // K - Maximum realistic temperature
#define MAX_MAGNETIC_FIELD       100.0     // nT - Maximum realistic field
#define MIN_VALID_MEASUREMENTS   3         // Minimum valid measurements

// Alert thresholds
#define CRITICAL_CME_THRESHOLD   0.8       // Critical CME confidence threshold
#define WARNING_CME_THRESHOLD    0.6       // Warning CME confidence threshold

// Prediction model parameters
#define TREND_DECAY_FACTOR       0.95      // Exponential decay for trend prediction
#define NOISE_LEVEL              0.05      // Noise level for predictions
#define SEASONAL_AMPLITUDE       0.1       // Seasonal variation amplitude

// Multi-instrument fusion weights
#define SWIS_WEIGHT              0.4       // Weight for SWIS data
#define SOLERIOX_WEIGHT          0.3       // Weight for SOLERIOX data
#define MAGNETOMETER_WEIGHT      0.3       // Weight for magnetometer data

// Data validation ranges
#define MIN_SOLAR_WIND_SPEED     200.0     // km/s
#define MIN_PROTON_DENSITY       0.1       // cm⁻³
#define MIN_TEMPERATURE          1000.0    // K
#define MIN_MAGNETIC_FIELD       0.1       // nT

#endif // MODEL_WEIGHTS_H
