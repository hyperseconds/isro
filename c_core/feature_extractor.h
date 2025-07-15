/*
 * Feature Extraction Header for TITANUS CME Detection
 */

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <math.h>

// Function prototypes for feature extraction

/**
 * Extract gradient (slope) from time series data
 * @param values Array of values
 * @param count Number of values
 * @param time_interval Time interval between measurements
 * @return Gradient value
 */
double extract_gradient(double* values, int count, double time_interval);

/**
 * Extract delta (change) between first and last values
 * @param values Array of values
 * @param count Number of values
 * @return Delta value
 */
double extract_delta(double* values, int count);

/**
 * Extract variance from time series
 * @param values Array of values
 * @param count Number of values
 * @return Variance value
 */
double extract_variance(double* values, int count);

/**
 * Extract magnetic field rotation angle
 * @param bx X component of magnetic field
 * @param by Y component of magnetic field
 * @param bz Z component of magnetic field
 * @return Rotation angle in degrees
 */
double extract_magnetic_rotation(double bx, double by, double bz);

/**
 * Extract spectral power in frequency band
 * @param values Array of values
 * @param count Number of values
 * @param frequency_band_low Lower frequency bound
 * @param frequency_band_high Upper frequency bound
 * @return Spectral power
 */
double extract_spectral_power(double* values, int count, double frequency_band_low, double frequency_band_high);

/**
 * Calculate proton beta parameter
 * @param proton_density Proton density (cm⁻³)
 * @param temperature Temperature (K)
 * @param magnetic_field_magnitude Magnetic field magnitude (nT)
 * @return Beta parameter
 */
double calculate_proton_beta(double proton_density, double temperature, double magnetic_field_magnitude);

/**
 * Calculate Alfvén speed
 * @param proton_density Proton density (cm⁻³)
 * @param magnetic_field_magnitude Magnetic field magnitude (nT)
 * @return Alfvén speed (km/s)
 */
double calculate_alfven_speed(double proton_density, double magnetic_field_magnitude);

/**
 * Detect particle flux dropout
 * @param ion_flux_values Array of ion flux values
 * @param count Number of values
 * @return Dropout ratio (0-1)
 */
double detect_flux_dropout(double* ion_flux_values, int count);

/**
 * Detect bidirectional electron beams
 * @param electron_flux_parallel Parallel electron flux
 * @param electron_flux_antiparallel Antiparallel electron flux
 * @param count Number of measurements
 * @return 1 if bidirectional, 0 otherwise
 */
int detect_bidirectional_electrons(double* electron_flux_parallel, double* electron_flux_antiparallel, int count);

/**
 * Calculate solar wind dynamic pressure
 * @param proton_density Proton density (cm⁻³)
 * @param solar_wind_speed Solar wind speed (km/s)
 * @return Dynamic pressure (nPa)
 */
double calculate_solar_wind_dynamic_pressure(double proton_density, double solar_wind_speed);

#endif // FEATURE_EXTRACTOR_H
