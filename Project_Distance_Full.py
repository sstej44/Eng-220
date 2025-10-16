
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time

# Start timer for the entire project
start_time = time.time()

# --- Load Dataset ---
csv_path = "subset_100000_rows.csv"  # Update this path if needed
data = pd.read_csv(csv_path)

# --- Preprocessing ---
# Handle missing data with mean imputation
data = data.fillna(data.mean(numeric_only=True))

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# --- Task 1: Predictive Modeling ---
print("Task 1: Predictive Modeling...")

# Split into X and Y
X = normalized_data[:, :-1]
Y = normalized_data[:, -1]

# Train-test split (80-20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"\nTask 1 Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

task1_time = time.time() - start_time
print(f"Task 1 Execution Time: {task1_time:.2f} seconds")

# --- Task 2: Anomaly Detection ---
task2_start = time.time()
print("\nTask 2: Anomaly Detection...")

# Parameters for k-NN Distance
k = 3  # Number of neighbors
threshold_percentile = 60  # Outlier threshold percentile

# Compute k-NN distances
nbrs = NearestNeighbors(n_neighbors=k+1).fit(normalized_data)
distances, indices = nbrs.kneighbors(normalized_data)

# Exclude self-distance (first column)
distances = distances[:, 1:]

# Compute average k-NN distance for each point
k_distances = distances.mean(axis=1)

# Determine outliers based on a threshold
threshold = np.percentile(k_distances, threshold_percentile)
anomalies = k_distances > threshold

# Evaluation (simplified: same as MATLAB)
precision = np.sum(anomalies) / len(anomalies)
recall = precision
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(data["coolant"], data["stator_winding"], c=anomalies, cmap="bwr", marker="o")
plt.xlabel("Coolant Temperature")
plt.ylabel("Stator Winding Temperature")
plt.title("Anomaly Detection in Motor Temperature (k-NN Distance-Based)")
plt.grid(True)
plt.legend(["Normal", "Anomaly"])
plt.show()

# Task 2 timing
task2_time = time.time() - task2_start
print(f"Task 2 Execution Time: {task2_time:.2f} seconds")

# --- Total Execution Time ---
total_time = time.time() - start_time
print(f"\nTotal Execution Time: {total_time:.2f} seconds")
