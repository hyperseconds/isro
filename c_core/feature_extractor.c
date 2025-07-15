/*
 * Feature Extraction Functions for TITANUS CME Detection
 */

#include <math.h>
#include <stdio.h>
#include "feature_extractor.h"

double extract_gradient(double* values, int count, double time_interval) {
    if (count < 2) return 0.0;
    
    // Simple linear gradient calculation
    double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0;
    
    for (int i = 0; i < count; i++) {
        double x = i * time_interval;
        double y = values[i];
        
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
    }
    
    double denominator = count * sum_x2 - sum_x * sum_x;
    if (fabs(denominator) < 1e-10) return 0.0;
    
    return (count * sum_xy - sum_x * sum_y) / denominator;
}

double extract_delta(double* values, int count) {
    if (count < 2) return 0.0;
    
    return values[count - 1] - values[0];
}

double extract_variance(double* values, int count) {
    if (count < 2) return 0.0;
    
    double mean = 0.0;
    for (int i = 0; i < count; i++) {
        mean += values[i];
    }
    mean /= count;
    
    double variance = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = values[i] - mean;
        variance += diff * diff;
    }
    
    return variance / (count - 1);
}

double extract_magnetic_rotation(double bx, double by, double bz) {
    // Calculate rotation angle in magnetic field vector
    // This is a simplified calculation - in reality, this would require
    // time series of magnetic field vectors
    
    double magnitude = sqrt(bx*bx + by*by + bz*bz);
    if (magnitude < 1e-10) return 0.0;
    
    // Calculate deviation from expected quiet-time field direction
    // Assuming quiet-time field is primarily in X direction
    double expected_bx = magnitude;
    double expected_by = 0.0;
    double expected_bz = 0.0;
    
    // Calculate angle between current and expected field
    double dot_product = bx * expected_bx + by * expected_by + bz * expected_bz;
    double cos_angle = dot_product / (magnitude * magnitude);
    
    // Clamp to valid range for acos
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;
    
    return acos(cos_angle) * 180.0 / M_PI; // Return angle in degrees
}

double extract_spectral_power(double* values, int count, double frequency_band_low, double frequency_band_high) {
    // Simplified spectral power calculation
    // In a real implementation, this would use FFT
    
    if (count < 4) return 0.0;
    
    double power = 0.0;
    
    // Simple approximation: calculate power in time domain
    for (int i = 1; i < count; i++) {
        double derivative = values[i] - values[i-1];
        power += derivative * derivative;
    }
    
    return power / (count - 1);
}

double calculate_proton_beta(double proton_density, double temperature, double magnetic_field_magnitude) {
    // Calculate plasma beta parameter
    // β = (n * k * T) / (B² / 2μ₀)
    
    const double k_boltzmann = 1.38e-23; // J/K
    const double mu_0 = 4e-7 * M_PI;     // H/m
    const double nT_to_T = 1e-9;         // Convert nT to T
    
    if (magnetic_field_magnitude < 1e-10) return 0.0;
    
    double thermal_pressure = proton_density * 1e6 * k_boltzmann * temperature; // Convert cm⁻³ to m⁻³
    double magnetic_pressure = (magnetic_field_magnitude * nT_to_T) * (magnetic_field_magnitude * nT_to_T) / (2.0 * mu_0);
    
    return thermal_pressure / magnetic_pressure;
}

double calculate_alfven_speed(double proton_density, double magnetic_field_magnitude) {
    // Calculate Alfvén speed: v_A = B / √(μ₀ * ρ)
    
    const double mu_0 = 4e-7 * M_PI;     // H/m
    const double proton_mass = 1.67e-27; // kg
    const double nT_to_T = 1e-9;         // Convert nT to T
    
    if (proton_density < 1e-10) return 0.0;
    
    double mass_density = proton_density * 1e6 * proton_mass; // Convert cm⁻³ to m⁻³
    double B_tesla = magnetic_field_magnitude * nT_to_T;
    
    return B_tesla / sqrt(mu_0 * mass_density) / 1000.0; // Convert m/s to km/s
}

double detect_flux_dropout(double* ion_flux_values, int count) {
    // Detect sudden dropout in particle flux (typical CME signature)
    
    if (count < 3) return 0.0;
    
    double baseline = 0.0;
    int baseline_count = count / 3; // Use first third as baseline
    
    for (int i = 0; i < baseline_count; i++) {
        baseline += ion_flux_values[i];
    }
    baseline /= baseline_count;
    
    // Look for significant drops in the remaining data
    double max_dropout = 0.0;
    for (int i = baseline_count; i < count; i++) {
        if (baseline > 0) {
            double dropout_ratio = (baseline - ion_flux_values[i]) / baseline;
            if (dropout_ratio > max_dropout) {
                max_dropout = dropout_ratio;
            }
        }
    }
    
    return max_dropout;
}

int detect_bidirectional_electrons(double* electron_flux_parallel, double* electron_flux_antiparallel, int count) {
    // Detect bidirectional electron beams (strong CME indicator)
    
    if (count < 2) return 0;
    
    double parallel_mean = 0.0, antiparallel_mean = 0.0;
    
    for (int i = 0; i < count; i++) {
        parallel_mean += electron_flux_parallel[i];
        antiparallel_mean += electron_flux_antiparallel[i];
    }
    
    parallel_mean /= count;
    antiparallel_mean /= count;
    
    // Check if both directions show enhanced flux
    double enhancement_threshold = 1.5; // 50% above background
    
    return (parallel_mean > enhancement_threshold && antiparallel_mean > enhancement_threshold) ? 1 : 0;
}

double calculate_solar_wind_dynamic_pressure(double proton_density, double solar_wind_speed) {
    // Calculate dynamic pressure: P_dyn = ρ * v²
    // Where ρ = n * m_p (mass density)
    
    const double proton_mass = 1.67e-27; // kg
    const double km_to_m = 1000.0;
    
    double mass_density = proton_density * 1e6 * proton_mass; // Convert cm⁻³ to m⁻³
    double speed_ms = solar_wind_speed * km_to_m; // Convert km/s to m/s
    
    return mass_density * speed_ms * speed_ms * 1e9; // Convert to nPa
}
