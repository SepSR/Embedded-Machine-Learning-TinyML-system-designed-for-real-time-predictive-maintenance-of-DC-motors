/**
 * file: TinyML_Model_Run.ino
 * author: Sepehr
 * brief: Embedded Real-Time Predictive Maintenance Engine for ESP32.
 * 
 * This firmware acts as the main application loop. It implements:
 * 1. 10-stage Soft-Start PWM DC motor speed control to limit current spikes (< 500mA).
 * 2. Real-time statistical feature extraction over 500ms windows.
 * 3. Offline-trained local SVM classifier inference on the edge.
 * 4. Temporal sliding window filter to verify persistent faults before sounding alarms.
 * 5. Landscape 320x240 screen rendering with high-contrast, zero-flicker updates.
 */

#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>
#include "model.h" // Local SVM Classifier exported from the Python pipeline

// Unlock the namespace used by micromlgen to expose model variables
using namespace Eloquent::ML::Port;

// ST7789 SPI Pin Mapping (Flipped 180-degrees Landscape profile)
#define TFT_CS    5
#define TFT_DC    2
#define TFT_RST   4

Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

// IO Port Pin Configurations
#define SW420_PIN   25
#define MOTOR_PIN1  12
#define MOTOR_PIN2  13
#define BUZZER_PIN  26
#define BUTTON_PIN  27

// Asynchronous sensor feature tracking registers
volatile unsigned long pulse_count = 0;
volatile unsigned long last_pulse_time = 0;
volatile unsigned long max_gap = 0;
volatile unsigned long active_time = 0;

// Motor state machine parameters
int motor_speed_level = 1; // System starts automatically at Level 1 to prevent standstill spikes
int current_pwm = 0;       // Monitors the actively applied PWM duty cycle
const int TOTAL_SPEED_STAGES = 10;
const int MIN_SAFE_PWM = 50;  // Delivers high starting torque
const int MAX_SAFE_PWM = 77;  // Upper bound to prevent current from exceeding safe thresholds

// Button debouncing registers
bool last_button_state = HIGH;
unsigned long last_debounce_time = 0;
const unsigned long DEBOUNCE_DELAY = 50; 

// Alarm filtering configuration (prevents transient noise from causing false alarms)
int anomaly_consecutive_count = 0;
const int ANOMALY_THRESHOLD_LIMIT = 3; // Requires 3 consecutive anomaly frames to sound alarm

// Instantiate the Machine Learning Classifier
VibrationClassifier classifier;

/**
 * @brief High-speed ISR to log pulse counts and calculate peak vibration intervals
 */
void IRAM_ATTR sensorISR() {
  unsigned long now = micros();
  pulse_count = pulse_count + 1;
  
  if (last_pulse_time > 0) {
    unsigned long gap = now - last_pulse_time;
    if (gap > max_gap) {
      max_gap = gap;
    }
  }
  
  active_time = active_time + 100;
  last_pulse_time = now;
}

// GUI Drawing Prototypes
void drawStaticUI();
void updateMotorSpeedUI(int level);
void updateSensorDataUI(unsigned long pulses, float ratio);
void triggerAlarmUI(bool anomaly);

/**
 * @brief Soft-Start function to ramp up motor speed gradually.
 * 
 * Stepping the PWM signal slowly over a 100ms window prevents massive back-EMF spikes 
 * and inductive inrush currents, completely eliminating system power drops and ESP32 resets.
 */
void setMotorSpeedSoft(int target_pwm) {
  if (current_pwm == target_pwm) return;
  
  int step = (target_pwm > current_pwm) ? 1 : -1;
  
  while (current_pwm != target_pwm) {
    current_pwm = current_pwm + step;
    analogWrite(MOTOR_PIN1, current_pwm);
    delay(4); // 4ms delay per step provides a gentle ramp-up curve
  }
  digitalWrite(MOTOR_PIN2, LOW);
}

void setup() {
  // Pin directions
  pinMode(SW420_PIN, INPUT_PULLUP);
  pinMode(MOTOR_PIN1, OUTPUT);
  pinMode(MOTOR_PIN2, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize TFT Display
  tft.init(240, 320); 
  tft.setRotation(3); // Rotate 180 degrees to face the operator directly
  tft.fillScreen(ST77XX_BLACK);
  
  // Draw layout
  drawStaticUI();
  
  // Slowly spin-up motor to Level 1 during startup
  setMotorSpeedSoft(MIN_SAFE_PWM);
  updateMotorSpeedUI(motor_speed_level);
  triggerAlarmUI(false);

  // Activate interrupts after safe startup
  attachInterrupt(digitalPinToInterrupt(SW420_PIN), sensorISR, RISING);
}

void loop() {
  // 1. Handle Debounced Button Press to Cycle 10 Speed Stages
  int reading = digitalRead(BUTTON_PIN);
  if (reading != last_button_state) {
    last_debounce_time = millis();
  }
  
  if ((millis() - last_debounce_time) > DEBOUNCE_DELAY) {
    if (reading == LOW) {
      // Loop from 1 up to 10
      motor_speed_level = (motor_speed_level % TOTAL_SPEED_STAGES) + 1;
      
      // Calculate targeted PWM duty cycle
      int target_pwm = MIN_SAFE_PWM + (motor_speed_level - 1) * (MAX_SAFE_PWM - MIN_SAFE_PWM) / (TOTAL_SPEED_STAGES - 1);
      
      // Ramp motor speed smoothly
      setMotorSpeedSoft(target_pwm);
      updateMotorSpeedUI(motor_speed_level);
      
      delay(200); // Prevent multi-trigger inputs
    }
  }
  last_button_state = reading;

  // 2. Sample window evaluation & TinyML inference
  delay(500); 
  
  // Critical section: detach interrupt to perform calculations
  detachInterrupt(digitalPinToInterrupt(SW420_PIN));
  
  float active_ratio = (float)active_time / 500000.0;
  if (active_ratio > 1.0) active_ratio = 1.0;
  
  // Generate inputs array
  float features[3] = { (float)pulse_count, active_ratio, (float)max_gap };
  
  // Call internal SVM classification function
  int prediction = classifier.predict(features);
  
  // Render live variables
  updateSensorDataUI(pulse_count, active_ratio);
  
  // Apply temporal verification filter to clear sporadic mechanical bounce noise
  if (prediction == 1) { 
    anomaly_consecutive_count++;
  } else {
    if (anomaly_consecutive_count > 0) {
      anomaly_consecutive_count--; 
    }
  }

  // Actuate buzzer and display state
  if (anomaly_consecutive_count >= ANOMALY_THRESHOLD_LIMIT) {
    triggerAlarmUI(true);  
  } else {
    triggerAlarmUI(false); 
  }
  
  // Reset sampling registers
  pulse_count = 0;
  max_gap = 0;
  active_time = 0;
  last_pulse_time = micros();
  
  // Re-attach hardware interrupt to sample the next window
  attachInterrupt(digitalPinToInterrupt(SW420_PIN), sensorISR, RISING);
}

// --- Graphical Rendering Implementations (Flicker-Free Localized Redraw) ---

void drawStaticUI() {
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);
  
  tft.setCursor(10, 10);
  tft.println("TINYML MOTOR MONITOR");
  tft.drawFastHLine(10, 30, 300, ST77XX_BLUE);
  
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(15, 50);
  tft.print("Motor Mode: ");
  
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(15, 95);
  tft.print("Vib Pulses: ");
  
  tft.setCursor(15, 135);
  tft.print("Active Ratio: ");
  
  tft.setCursor(15, 185);
  tft.print("Diag. Status: ");
}

void updateMotorSpeedUI(int level) {
  tft.fillRect(150, 50, 160, 20, ST77XX_BLACK);
  tft.setCursor(150, 50);
  tft.setTextSize(2);
  
  tft.setTextColor(ST77XX_GREEN);
  tft.print(level);
  tft.print(" / ");
  tft.print(TOTAL_SPEED_STAGES);
}

void updateSensorDataUI(unsigned long pulses, float ratio) {
  tft.fillRect(150, 95, 160, 20, ST77XX_BLACK);
  tft.fillRect(170, 135, 140, 20, ST77XX_BLACK);
  
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_CYAN);
  
  tft.setCursor(150, 95);
  tft.print(pulses);
  
  tft.setCursor(170, 135);
  tft.print(ratio, 4);
}

void triggerAlarmUI(bool anomaly) {
  tft.fillRect(170, 185, 140, 25, ST77XX_BLACK);
  tft.setCursor(170, 185);
  tft.setTextSize(2);
  
  if (anomaly) {
    tft.setTextColor(ST77XX_RED);
    tft.print("FAULT!");
    tone(BUZZER_PIN, 1200); 
  } else {
    tft.setTextColor(ST77XX_GREEN);
    tft.print("HEALTHY");
    noTone(BUZZER_PIN); 
  }
}