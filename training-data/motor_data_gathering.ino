/**
 * file: motor_data_gathering.ino
 * author: SepSR
 * brief: High-frequency feature extraction firmware for vibration data collection.
 * 
 * This program interfaces with a 2-pin SW-420 mechanical vibration switch to 
 * transform rapid digital pulse streams into temporal statistical features. 
 * The data is formatted as CSV and outputted over the serial interface to build 
 * the training dataset for the TinyML model.
 */

#include <Arduino.h>

// Hardware pin assignments
#define SW420_PIN 25
#define MOTOR_PIN1 12
#define MOTOR_PIN2 13
#define BUTTON_PIN 27

// Interrupt-safe variables for temporal feature tracking
volatile unsigned long pulse_count = 0;
volatile unsigned long last_pulse_time = 0;
volatile unsigned long max_gap = 0;
volatile unsigned long active_time = 0;

/**
 * @brief Interrupt Service Routine (ISR) triggered on the rising edge of the SW-420.
 * 
 * Since the raw sensor acts as a Normally Closed (NC) switch, a physical vibration 
 * temporarily breaks the circuit, causing the internal pull-up resistor to pull 
 * the GPIO pin HIGH. We capture this transition to measure vibration frequency and patterns.
 */
void IRAM_ATTR sensorISR() {
  unsigned long now = micros();
  
  // Standard compliant assignment to prevent C++20 volatile increment warnings
  pulse_count = pulse_count + 1;
  
  // Calculate the time interval (gap) between consecutive vibration pulses
  if (last_pulse_time > 0) {
    unsigned long gap = now - last_pulse_time;
    if (gap > max_gap) {
      max_gap = gap;
    }
  }
  
  // Estimate total active vibration duration (assuming a 100-microsecond pulse width)
  active_time = active_time + 100; 
  last_pulse_time = now;
}

void setup() {
  // Initialize Serial interface at a high baud rate to prevent output bottlenecks
  Serial.begin(115200);
  
  // Configure peripherals
  pinMode(SW420_PIN, INPUT_PULLUP);
  pinMode(MOTOR_PIN1, OUTPUT);
  pinMode(MOTOR_PIN2, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Register the high-speed ISR to handle sensor pulse transitions
  attachInterrupt(digitalPinToInterrupt(SW420_PIN), sensorISR, RISING);
  
  // Spin the motor at a safe, low speed to collect baseline/normal data
  analogWrite(MOTOR_PIN1, 75); 
  digitalWrite(MOTOR_PIN2, LOW);
  
  // Print CSV header schema for the data logging software
  Serial.println("PulseCount,ActiveRatio,MaxGap");
}

void loop() {
  // Samples and prints the extracted features every 500 milliseconds
  delay(500);
  
  // Temporarily disable interrupts to safely read volatile registers without race conditions
  detachInterrupt(digitalPinToInterrupt(SW420_PIN));
  
  // Compute the duty cycle ratio of active vibration to the total sampling window (500,000 microseconds)
  float active_ratio = (float)active_time / 500000.0;
  if (active_ratio > 1.0) active_ratio = 1.0;
  
  // Output the formatted CSV record
  Serial.print(pulse_count);
  Serial.print(",");
  Serial.print(active_ratio, 4);
  Serial.print(",");
  Serial.println(max_gap);
  
  // Reset all registers to prepare for the next 500ms sliding window
  pulse_count = 0;
  max_gap = 0;
  active_time = 0;
  last_pulse_time = micros();
  
  // Re-enable the hardware interrupt
  attachInterrupt(digitalPinToInterrupt(SW420_PIN), sensorISR, RISING);
}