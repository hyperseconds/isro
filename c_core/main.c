/*
 * TITANUS CME Prediction Engine - Main C Program
 * Reads fused_input.json, processes features, writes prediction_output.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cJSON.h"
#include "feature_extractor.h"
#include "model_weights.h"

typedef struct {
    double solar_wind_speed;
    double proton_density;
    double temperature;
    double ion_flux;
    double electron_flux;
    double magnetic_field_x;
    double magnetic_field_y;
    double magnetic_field_z;
    double magnetic_field_magnitude;
    double dynamic_pressure;
} FeatureVector;

typedef struct {
    int cme_detected;
    double confidence;
    FeatureVector extracted_features;
    double future_predictions[PREDICTION_HOURS][NUM_PREDICTION_PARAMS];
} PredictionResult;

// Function prototypes
char* read_file(const char* filename);
int write_file(const char* filename, const char* content);
FeatureVector extract_features_from_json(cJSON* json);
PredictionResult predict_cme(FeatureVector features);
void generate_future_predictions(FeatureVector current_features, double predictions[][NUM_PREDICTION_PARAMS]);
double calculate_confidence(FeatureVector features, int cme_detected);
char* create_output_json(PredictionResult result);

int main() {
    printf("TITANUS CME Prediction Engine Starting...\n");
    
    // Read input JSON file
    char* input_json_str = read_file("../fused/fused_input.json");
    if (!input_json_str) {
        fprintf(stderr, "Error: Cannot read fused_input.json\n");
        return 1;
    }
    
    // Parse JSON
    cJSON* json = cJSON_Parse(input_json_str);
    if (!json) {
        fprintf(stderr, "Error: Invalid JSON format\n");
        free(input_json_str);
        return 1;
    }
    
    // Extract features
    FeatureVector features = extract_features_from_json(json);
    
    // Run prediction
    PredictionResult result = predict_cme(features);
    
    // Create output JSON
    char* output_json_str = create_output_json(result);
    if (!output_json_str) {
        fprintf(stderr, "Error: Failed to create output JSON\n");
        cJSON_Delete(json);
        free(input_json_str);
        return 1;
    }
    
    // Write output file
    if (write_file("../fused/prediction_output.json", output_json_str) != 0) {
        fprintf(stderr, "Error: Cannot write prediction_output.json\n");
        free(output_json_str);
        cJSON_Delete(json);
        free(input_json_str);
        return 1;
    }
    
    printf("CME Prediction completed successfully\n");
    printf("CME Detected: %s\n", result.cme_detected ? "YES" : "NO");
    printf("Confidence: %.3f\n", result.confidence);
    
    // Cleanup
    free(output_json_str);
    cJSON_Delete(json);
    free(input_json_str);
    
    return 0;
}

char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* content = malloc(length + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }
    
    fread(content, 1, length, file);
    content[length] = '\0';
    
    fclose(file);
    return content;
}

int write_file(const char* filename, const char* content) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        return -1;
    }
    
    fprintf(file, "%s", content);
    fclose(file);
    return 0;
}

FeatureVector extract_features_from_json(cJSON* json) {
    FeatureVector features = {0};
    
    cJSON* features_obj = cJSON_GetObjectItem(json, "features");
    if (!features_obj) {
        return features;
    }
    
    // Extract numerical features
    cJSON* item;
    
    item = cJSON_GetObjectItem(features_obj, "solar_wind_speed");
    if (cJSON_IsNumber(item)) features.solar_wind_speed = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "proton_density");
    if (cJSON_IsNumber(item)) features.proton_density = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "temperature");
    if (cJSON_IsNumber(item)) features.temperature = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "ion_flux");
    if (cJSON_IsNumber(item)) features.ion_flux = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "electron_flux");
    if (cJSON_IsNumber(item)) features.electron_flux = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "magnetic_field_x");
    if (cJSON_IsNumber(item)) features.magnetic_field_x = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "magnetic_field_y");
    if (cJSON_IsNumber(item)) features.magnetic_field_y = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "magnetic_field_z");
    if (cJSON_IsNumber(item)) features.magnetic_field_z = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "magnetic_field_magnitude");
    if (cJSON_IsNumber(item)) features.magnetic_field_magnitude = item->valuedouble;
    
    item = cJSON_GetObjectItem(features_obj, "dynamic_pressure");
    if (cJSON_IsNumber(item)) features.dynamic_pressure = item->valuedouble;
    
    return features;
}

PredictionResult predict_cme(FeatureVector features) {
    PredictionResult result = {0};
    result.extracted_features = features;
    
    // Apply threshold-based detection
    int detection_flags = 0;
    
    // Solar wind speed threshold
    if (features.solar_wind_speed > SOLAR_WIND_THRESHOLD) {
        detection_flags++;
    }
    
    // Proton density threshold
    if (features.proton_density > PROTON_DENSITY_THRESHOLD) {
        detection_flags++;
    }
    
    // Magnetic field rotation detection
    double mag_rotation = extract_magnetic_rotation(
        features.magnetic_field_x,
        features.magnetic_field_y,
        features.magnetic_field_z
    );
    if (mag_rotation > MAGNETIC_ROTATION_THRESHOLD) {
        detection_flags++;
    }
    
    // Dynamic pressure enhancement
    if (features.dynamic_pressure > DYNAMIC_PRESSURE_THRESHOLD) {
        detection_flags++;
    }
    
    // Particle flux enhancement
    if (features.ion_flux > ION_FLUX_THRESHOLD || 
        features.electron_flux > ELECTRON_FLUX_THRESHOLD) {
        detection_flags++;
    }
    
    // Temperature depression (typical CME signature)
    if (features.temperature < TEMPERATURE_THRESHOLD) {
        detection_flags++;
    }
    
    // CME detection logic: require multiple indicators
    result.cme_detected = (detection_flags >= MIN_DETECTION_FLAGS);
    
    // Calculate confidence based on number of triggered thresholds
    result.confidence = calculate_confidence(features, result.cme_detected);
    
    // Generate future predictions
    generate_future_predictions(features, result.future_predictions);
    
    return result;
}

void generate_future_predictions(FeatureVector current_features, double predictions[][NUM_PREDICTION_PARAMS]) {
    // Simple time-series extrapolation for future predictions
    // In a real system, this would use sophisticated models
    
    for (int hour = 0; hour < PREDICTION_HOURS; hour++) {
        double time_factor = (hour + 1) / 24.0; // Normalize to 0-1 over 24 hours
        
        // Solar wind speed prediction (with decay if CME detected)
        predictions[hour][0] = current_features.solar_wind_speed * 
                              (1.0 - time_factor * 0.1);
        
        // Proton density prediction
        predictions[hour][1] = current_features.proton_density * 
                              (1.0 + sin(time_factor * M_PI) * 0.2);
        
        // Magnetic field magnitude prediction
        predictions[hour][2] = current_features.magnetic_field_magnitude * 
                              (1.0 + cos(time_factor * M_PI * 2) * 0.15);
        
        // Dynamic pressure prediction
        predictions[hour][3] = current_features.dynamic_pressure * 
                              (1.0 - time_factor * 0.05);
    }
}

double calculate_confidence(FeatureVector features, int cme_detected) {
    if (!cme_detected) {
        return 0.1; // Low confidence for no detection
    }
    
    double confidence = 0.0;
    int factor_count = 0;
    
    // Calculate confidence based on how much thresholds are exceeded
    if (features.solar_wind_speed > SOLAR_WIND_THRESHOLD) {
        confidence += (features.solar_wind_speed / SOLAR_WIND_THRESHOLD - 1.0) * 0.2;
        factor_count++;
    }
    
    if (features.proton_density > PROTON_DENSITY_THRESHOLD) {
        confidence += (features.proton_density / PROTON_DENSITY_THRESHOLD - 1.0) * 0.2;
        factor_count++;
    }
    
    if (features.dynamic_pressure > DYNAMIC_PRESSURE_THRESHOLD) {
        confidence += (features.dynamic_pressure / DYNAMIC_PRESSURE_THRESHOLD - 1.0) * 0.2;
        factor_count++;
    }
    
    // Normalize confidence
    if (factor_count > 0) {
        confidence = confidence / factor_count + 0.5; // Base confidence of 0.5
        confidence = fmin(confidence, 1.0); // Cap at 1.0
    } else {
        confidence = 0.5; // Default confidence
    }
    
    return confidence;
}

char* create_output_json(PredictionResult result) {
    cJSON* json = cJSON_CreateObject();
    cJSON* features = cJSON_CreateObject();
    cJSON* future_pred = cJSON_CreateObject();
    cJSON* thresholds = cJSON_CreateObject();
    
    // Basic prediction results
    cJSON_AddBoolToObject(json, "cme_detected", result.cme_detected);
    cJSON_AddNumberToObject(json, "confidence", result.confidence);
    cJSON_AddStringToObject(json, "prediction_method", "C_THRESHOLD_ENGINE");
    
    // Current time
    time_t now;
    time(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", gmtime(&now));
    cJSON_AddStringToObject(json, "timestamp", timestamp);
    
    // Extracted features
    cJSON_AddNumberToObject(features, "solar_wind_speed", result.extracted_features.solar_wind_speed);
    cJSON_AddNumberToObject(features, "proton_density", result.extracted_features.proton_density);
    cJSON_AddNumberToObject(features, "temperature", result.extracted_features.temperature);
    cJSON_AddNumberToObject(features, "ion_flux", result.extracted_features.ion_flux);
    cJSON_AddNumberToObject(features, "electron_flux", result.extracted_features.electron_flux);
    cJSON_AddNumberToObject(features, "magnetic_field_magnitude", result.extracted_features.magnetic_field_magnitude);
    cJSON_AddNumberToObject(features, "dynamic_pressure", result.extracted_features.dynamic_pressure);
    
    cJSON_AddItemToObject(json, "features", features);
    
    // Future predictions
    cJSON* speed_pred = cJSON_CreateArray();
    cJSON* density_pred = cJSON_CreateArray();
    cJSON* mag_pred = cJSON_CreateArray();
    cJSON* pressure_pred = cJSON_CreateArray();
    
    for (int i = 0; i < PREDICTION_HOURS; i++) {
        cJSON_AddItemToArray(speed_pred, cJSON_CreateNumber(result.future_predictions[i][0]));
        cJSON_AddItemToArray(density_pred, cJSON_CreateNumber(result.future_predictions[i][1]));
        cJSON_AddItemToArray(mag_pred, cJSON_CreateNumber(result.future_predictions[i][2]));
        cJSON_AddItemToArray(pressure_pred, cJSON_CreateNumber(result.future_predictions[i][3]));
    }
    
    cJSON_AddItemToObject(future_pred, "solar_wind_speed", speed_pred);
    cJSON_AddItemToObject(future_pred, "proton_density", density_pred);
    cJSON_AddItemToObject(future_pred, "magnetic_field_magnitude", mag_pred);
    cJSON_AddItemToObject(future_pred, "dynamic_pressure", pressure_pred);
    
    cJSON_AddItemToObject(json, "future_predictions", future_pred);
    
    // Thresholds used
    cJSON_AddNumberToObject(thresholds, "SOLAR_WIND_THRESHOLD", SOLAR_WIND_THRESHOLD);
    cJSON_AddNumberToObject(thresholds, "PROTON_DENSITY_THRESHOLD", PROTON_DENSITY_THRESHOLD);
    cJSON_AddNumberToObject(thresholds, "DYNAMIC_PRESSURE_THRESHOLD", DYNAMIC_PRESSURE_THRESHOLD);
    cJSON_AddNumberToObject(thresholds, "ION_FLUX_THRESHOLD", ION_FLUX_THRESHOLD);
    cJSON_AddNumberToObject(thresholds, "MIN_DETECTION_FLAGS", MIN_DETECTION_FLAGS);
    
    cJSON_AddItemToObject(json, "thresholds_used", thresholds);
    
    char* json_string = cJSON_Print(json);
    cJSON_Delete(json);
    
    return json_string;
}
