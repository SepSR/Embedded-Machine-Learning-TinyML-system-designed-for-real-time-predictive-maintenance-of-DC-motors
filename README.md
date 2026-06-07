# TinyML-Predictive-Maintenance: Edge-Based Anomaly Detection for DC Motors Using Local Support Vector Machines on ESP32

This repository presents an end-to-end, academic-grade implementation of an embedded machine learning (TinyML) system engineered for real-time predictive maintenance of industrial DC motors. Running completely offline and on-the-edge of an ESP32 microcontroller, the system continuously analyzes electromechanical vibrations to classify normal states and anomalous failures using an on-chip **Support Vector Machine (SVM)** classifier. 

To address practical constraints in industrial microcontrollers, this project implements **Inrush Current Soft-Starting** and **Temporal Confidence Filtering** to secure hardware integrity and eliminate false positive alarm triggers.

---

## 📊 Machine Learning Performance & Evaluations

The Support Vector Machine decision boundary and dataset distributions are mathematically modeled and visualized below. These plots demonstrate the classification capabilities of our edge engine.

### 1. 3D Feature Space Distribution
The electromechanical vibration features are plotted in 3D space (`PulseCount`, `ActiveRatio`, and `MaxGap`). This visualizes how the normal and anomalous physical operating states form highly separable clusters.

<img width="1941" height="2006" alt="feature_distribution_3d" src="https://github.com/user-attachments/assets/29722af4-1d4f-4aff-a571-1fdb384db78c" />


### 2. 2D Classification Boundary Plane
This plot represents the trained linear decision boundary of the SVM classifier projected onto a 2D plane. The decision boundary demonstrates how the model segregates nominal vibrations from structural faults.

<img width="2536" height="1616" alt="svm_decision_boundary" src="https://github.com/user-attachments/assets/119bca5b-2e7d-49ba-920a-6ef395c0c7fb" />


### 3. Confusion Matrix and Receiver Operating Characteristic (ROC)
The confusion matrix verifies class prediction metrics on the test subset, while the ROC curve demonstrates excellent classification sensitivity with a high Area Under the Curve (AUC) rating.

<img width="4038" height="1762" alt="confusion_matrix_and_roc" src="https://github.com/user-attachments/assets/0044f0e4-f90f-4db2-ba7b-1e0ae884c253" />


---

## 🛠️ Key Engineering Techniques Implemented

### 1. Temporal Feature Engineering (On-the-Edge)
Instead of feeding raw, high-frequency digital pulse sequences into the classifier, the system performs edge-based feature extraction. Over a **500ms sliding temporal window**, three distinct statistical features are dynamically calculated:
*   **Vibration Pulse Count:** Represents the mechanical vibration frequency.
*   **Vibration Active Ratio:** The percentage of time the mechanical switch spent in the open (vibrating) state.
*   **Maximum Gap Interval:** The maximum time interval between consecutive pulses, representing the rhythm/symmetry of the vibration.

### 2. Embedded SVM Inference (TinyML Model)
A **Linear Support Vector Classifier (SVC)** was trained on collected healthy and faulty datasets. To maintain compliance with resource-constrained MCUs, the model's multidimensional decision boundaries were compiled directly into dependency-free, optimized static C++ code using `micromlgen`. This eliminates OS and library overhead, executing local predictions in microseconds.

### 3. Inductive Load Soft-Start PWM Control
Directly starting high-current DC motors from a standstill causes a massive inrush current (stall current spike) that triggers voltage drops and brownout resets on the ESP32. We resolved this by implementing a **Software Soft-Start algorithm** that ramps up the PWM duty cycle linearly over a 100ms window, smoothing the current gradient ($dI/dt$) and ensuring complete power supply stability.

### 4. Consecutive-Anomaly Confidence Window
To suppress mechanical contact bounce and electrical noise, a **Sliding Window Temporal Filter** is implemented. Rather than triggering the fault alarm on a single positive prediction, the system requires a minimum of **3 consecutive anomalous classifications** before declaring a system fault, completely eliminating false positives.


---

## 🔌 Hardware Schematics and Pin Connections

The system is powered by an external 5V 2A adapter. A linear low-dropout (LDO) regulator steps down the voltage to 3.3V to power the microcontroller and sensor logic. Decoupling capacitor pairs are integrated to suppress high-frequency inductive feedback.

### Complete Wiring Map

| Component | Pin Label | ESP32 Pin / Power Source | Description |
| :--- | :--- | :--- | :--- |
| **ST7789 TFT** | VCC | 3.3V (LDO Output) | Display Power |
| | GND | GND | Common Ground |
| | CS | GPIO5 | SPI Chip Select |
| | RESET | GPIO4 | SPI Hardware Reset |
| | DC/RS | GPIO2 | Data / Command Select |
| | SDI (MOSI)| GPIO23 | SPI MOSI Data Line |
| | SCK | GPIO18 | SPI Clock Line |
| | LED | 3.3V (LDO Output) | Backlight Power |
| **SW-420 Sensor**| Terminal 1| GND | Reference Ground |
| (Vibration Switch)| Terminal 2| GPIO25 | Input Interrupt Pin (Pull-up) |
| **DRV8833** | VM | 5V (Adapter Output) | High-Power Motor Rail |
| (Motor Driver) | GND | GND | Common Ground |
| | AIN1 | GPIO12 | PWM Speed Interface |
| | AIN2 | GPIO13 | Direction Controller (GND) |
| | STBY | 3.3V (LDO Output) | Enable Driver |
| | AO1 / AO2 | DC Motor Terminals | Active Motor Outputs |
| **Buzzer** | Positive (+)| GPIO26 | PWM Sound Output |
| | Negative (-)| GND | Common Ground |
| **Control Button**| Terminal 1| GPIO27 | Speed Level Switch |
| | Terminal 2| GND | Common Ground |


### ⚡ Decoupling Capacitor Configuration
*   **High-Power Rail (VM - GND):** A parallel combination of a **100uF Electrolytic capacitor** and a **10nF Ceramic capacitor (103)** is placed closest to the DRV8833 VM pin to filter out low-frequency and high-frequency noise spikes from the motor.
*   **Logic Rail (3.3V - GND):** A parallel combination of a **100uF Electrolytic capacitor** and a **10nF Ceramic capacitor (103)** is placed at the output of the LDO regulator to stabilize the ESP32 and LCD during motor state changes.

---

### 🖼️ Circuit Picture

<img width="894" height="1280" alt="image" src="https://github.com/user-attachments/assets/7889fe5b-5645-49fb-9ac1-28c5c03e8d82" />


### 📈 Live Data Collection Using Serial Monitor

<img width="903" height="719" alt="image" src="https://github.com/user-attachments/assets/fbc11621-5138-4a20-a64b-6a7a5b3b9feb" />

### 🎥 Running TinyML On ESP32

https://github.com/user-attachments/assets/ab03e6a6-1329-45a2-9edd-b3d7a31d29d0


---

## 🚀 Step-by-Step Setup and Deployment

### 1. Data Collection
1. Upload `motor_data_gathering.ino` to the ESP32.
2. Run the motor under **Normal (Healthy)** conditions. Copy the CSV output from the Serial Monitor and save it locally as `normal.csv`.
3. Introduce an unbalanced mass to the motor shaft (e.g., a small piece of electrical tape) to simulate an **Anomaly**. Save the CSV output as `anomaly.csv`.


### 2. Model Training & Export
1. Place `normal.csv` and `anomaly.csv` in the same directory as `train.py`.
2. Run the Python script:
   ```bash
   pip install numpy pandas scikit-learn micromlgen matplotlib
   python train.py
   The script will output the training accuracy, generate the diagnostic evaluation plots, and write the C++ model to model.h.
### 3. Edge Deployment
Move the generated model.h file to your TinyML_Model_Run Arduino project directory.
Open TinyML_Model_Run.ino in the Arduino IDE and compile/upload the code to your ESP32.
The system is now fully autonomous! Change the speed levels (1-10) using the control button and monitor the real-time system health on the ST7789 TFT display.

### 🔮 Future Research Directions & Architectural Extensions
To further scale this research in industrial settings, several advanced extensions are proposed:
1. Unsupervised Anomaly Detection (One-Class classification)
While the current supervised SVM classifier achieves high accuracy, industrial faults are diverse and often undocumented. Future iterations will transition to unsupervised algorithms, such as One-Class SVM or Autoencoders. By training solely on healthy baseline profiles, the edge system will flag any unseen, deviant behavior as anomalous, removing the requirement for hazardous fault simulation.
2. Decentralized Federated Learning
Deploying predictive maintenance across a fleet of industrial motors exposes data privacy and bandwidth constraints. Implementing Federated Learning on-chip will allow local nodes to train models independently and share only mathematical weight gradients with a centralized aggregator, continuously refining a global asset diagnostic model without transmitting raw mechanical vibration datasets.
3. Multi-Sensor Data Fusion
Integrating mechanical vibration signatures with secondary physical indicators (such as DHT11 ambient humidity/temperature and Motor Current Signature Analysis - MCSA via current shunts) will create a highly resilient multi-sensory diagnostic matrix, preventing false negatives induced by isolated environmental disturbances.
