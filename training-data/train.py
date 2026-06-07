"""
File: train.py
Author: Sepehr
Brief: Local Machine Learning pipeline to train an SVM classifier and export to C++.

This script parses 'normal.csv' and 'anomaly.csv', trains a Support Vector 
Classifier (SVC) with balanced weights, exports the decision boundary to a 
2D diagnostic plot, and compiles the model parameters into a C++ header file 
for ESP32 deployment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from micromlgen import port
import os

# Check if the generated CSV datasets exist in the current working directory
if not os.path.exists('normal.csv') or not os.path.exists('anomaly.csv'):
    print("Error: 'normal.csv' and 'anomaly.csv' must be placed in the current directory.")
    exit()

# 1. Load and parse dataset records
print("Loading CSV datasets...")
normal_df = pd.read_csv('normal.csv')
anomaly_df = pd.read_csv('anomaly.csv')

# Extract features into numerical NumPy arrays
X_normal = normal_df.values
X_anomaly = anomaly_df.values

# Generate labels: 0 for normal, 1 for structural/electrical anomalies
y_normal = np.zeros(X_normal.shape[0])
y_anomaly = np.ones(X_anomaly.shape[0])

# Concatenate arrays into training matrices
X = np.vstack((X_normal, X_anomaly))
y = np.hstack((y_normal, y_anomaly))

print(f"Total Normal samples: {len(X_normal)}")
print(f"Total Anomaly samples: {len(X_anomaly)}")

# 2. Fit the Support Vector Classifier
# We use a linear SVM. class_weight='balanced' prevents bias due to class size differences.
print("Fitting the Linear Support Vector Machine (SVM)...")
clf = SVC(kernel='linear', C=10.0, class_weight='balanced', gamma=0.001)
clf.fit(X, y)

accuracy = clf.score(X, y) * 100
print(f"SVM Model fit successfully. Training Accuracy: {accuracy:.2f}%")

# 3. Generate Diagnostic Machine Learning Plots
# We project the 3D feature space onto 2D (PulseCount vs ActiveRatio) to plot the decision boundary
print("Generating decision boundary visualization...")
plt.figure(figsize=(10, 6))

# Plot scatter points for both classes
plt.scatter(X_normal[:, 0], X_normal[:, 1], color='g', alpha=0.6, label='Healthy Operation')
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], color='r', alpha=0.6, label='Mechanical Anomaly')

# Calculate and overlay the decision boundary line: w0*x + w1*y + w2*z + b = 0
# For 2D plotting, we assume the third feature (MaxGap) is held at its median value
coef = clf.coef_[0]
intercept = clf.intercept_[0]
median_gap = np.median(X[:, 2])

# Rearranging: y = -(w0*x + w2*median_gap + b) / w1
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(coef[0] * x_vals + coef[2] * median_gap + intercept) / coef[1]

plt.plot(x_vals, y_vals, '--', color='blue', linewidth=2, label='SVM Decision Boundary')

plt.title('TinyML Motor Diagnostics - Support Vector Machine Decision Boundary', fontsize=12)
plt.xlabel('Vibration Pulse Count (500ms Window)', fontsize=10)
plt.ylabel('Vibration Active Ratio', fontsize=10)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

# Save the plot image for portfolio/GitHub documentation
plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
print("Plot saved successfully as 'svm_decision_boundary.png'.")

# 4. Export the classifier to dependency-free C++ code
print("Exporting model to C++ header file...")
cpp_code = port(clf, classname="VibrationClassifier")

# Fix the va_start GCC compiler warning automatically by patching the generated code
cpp_code = cpp_code.replace("va_start(w, 3);", "va_start(w, x);")

# Save code to header
with open("model.h", "w") as f:
    f.write(cpp_code)

print("C++ header 'model.h' written successfully!")