"""
File: train.py
Author: Sepehr
Brief: Academic TinyML pipeline for SVM training, C++ porting, and multi-view evaluation plotting.

This script parses 'normal.csv' and 'anomaly.csv', trains a Support Vector 
Classifier (SVC), generates three publication-ready diagnostic plots, and 
compiles the model into a standard C++ header file for ESP32.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from micromlgen import port
import os

# Set matplotlib style for academic plotting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Verify dataset existence
if not os.path.exists('normal.csv') or not os.path.exists('anomaly.csv'):
    print("Error: 'normal.csv' and 'anomaly.csv' must be located in the current working directory!")
    exit()

# 1. Dataset Loading and Feature Structuring
print("Parsing CSV data structures...")
normal_df = pd.read_csv('normal.csv')
anomaly_df = pd.read_csv('anomaly.csv')

X_normal = normal_df.values
X_anomaly = anomaly_df.values

y_normal = np.zeros(X_normal.shape[0])
y_anomaly = np.ones(X_anomaly.shape[0])

X = np.vstack((X_normal, X_anomaly))
y = np.hstack((y_normal, y_anomaly))

# Partition datasets into training and testing sets for objective evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Total samples: {X.shape[0]} | Features count: {X.shape[1]}")
print(f"Training subset shape: {X_train.shape[0]} | Testing subset shape: {X_test.shape[0]}")

# 2. Fit Support Vector Classifier with balanced class weights
print("Fitting the Support Vector Machine (SVM) model...")
clf = SVC(kernel='linear', C=10.0, class_weight='balanced', probability=True, gamma=0.001)
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train) * 100
test_acc = clf.score(X_test, y_test) * 100
print(f"Model fitted. Training Accuracy: {train_acc:.2f}% | Test Accuracy: {test_acc:.2f}%")

# -------------------------------------------------------------------------
# GRAPH 1: 3D Feature Space Scatter Plot
# -------------------------------------------------------------------------
print("Generating 3D feature space distribution plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_normal[:, 0], X_normal[:, 1], X_normal[:, 2], c='forestgreen', marker='o', alpha=0.6, label='Healthy Operation')
ax.scatter(X_anomaly[:, 0], X_anomaly[:, 1], X_anomaly[:, 2], c='crimson', marker='^', alpha=0.6, label='Mechanical Anomaly')

ax.set_title('3D Feature Space Clustering of Electromechanical Vibrations', fontsize=12, pad=15)
ax.set_xlabel('Vibration Pulse Count (500ms)', fontsize=10, labelpad=10)
ax.set_ylabel('Active Ratio', fontsize=10, labelpad=10)
ax.set_zlabel('Max Gap (microseconds)', fontsize=10, labelpad=10)
ax.legend(loc='best')
ax.view_init(elev=20, azim=45)

plt.savefig('feature_distribution_3d.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# GRAPH 2: 2D Decision Boundary (PulseCount vs ActiveRatio at Median MaxGap)
# -------------------------------------------------------------------------
print("Generating 2D decision boundary visualization...")
plt.figure(figsize=(10, 6))

plt.scatter(X_normal[:, 0], X_normal[:, 1], color='forestgreen', alpha=0.6, label='Healthy Operation')
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], color='crimson', alpha=0.6, label='Mechanical Anomaly')

coef = clf.coef_[0]
intercept = clf.intercept_[0]
median_gap = np.median(X[:, 2])

# Calculate decision boundary: w0*x + w1*y + w2*median_gap + b = 0 -> y = -(w0*x + w2*median_gap + b) / w1
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(coef[0] * x_vals + coef[2] * median_gap + intercept) / coef[1]

plt.plot(x_vals, y_vals, '--', color='royalblue', linewidth=2, label='SVM Decision Boundary')
plt.title('2D SVM Classification Boundary Plane (Temporal Projection)', fontsize=12)
plt.xlabel('Vibration Pulse Count (500ms Window)', fontsize=10)
plt.ylabel('Vibration Active Ratio', fontsize=10)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# GRAPH 3: Confusion Matrix & Receiver Operating Characteristic (ROC) Curve
# -------------------------------------------------------------------------
print("Generating Confusion Matrix and ROC Curve plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot A: Confusion Matrix Heatmap
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
disp.plot(cmap=plt.cm.Blues, ax=ax1, colorbar=False)
ax1.set_title('Test Set Confusion Matrix', fontsize=12)
ax1.grid(False)

# Subplot B: ROC Curve
y_probs = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=10)
ax2.set_ylabel('True Positive Rate', fontsize=10)
ax2.set_title('Receiver Operating Characteristic (ROC)', fontsize=12)
ax2.legend(loc="lower right")

plt.tight_layout()
plt.savefig('confusion_matrix_and_roc.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# 4. Port trained SVM model to standalone C++ code
# -------------------------------------------------------------------------
print("Exporting model to dependency-free C++...")
cpp_code = port(clf, classname="VibrationClassifier")

# Automatically fix the C++20 GCC varargs compilation warning
cpp_code = cpp_code.replace("va_start(w, 3);", "va_start(w, x);")

with open("model.h", "w") as f:
    f.write(cpp_code)

print("Diagnostic charts and compliant 'model.h' compiled successfully!")