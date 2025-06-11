# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# --- Load and Process Data ---
df = pd.read_csv('gestures.csv')
X = []
y = []

# Group data by gesture and recording session (approximated by gaps in timestamps)
# A simple way to identify separate recordings for the same gesture
df['time_diff'] = df.groupby('gesture')['timestamp'].diff().fillna(0)
df['recording_id'] = (df['time_diff'] > 1.0).cumsum()

grouped = df.groupby(['gesture', 'recording_id'])

print(f"Found {len(grouped)} total gesture recordings.")

for (gesture_name, rec_id), group in grouped:
    if len(group) < 10: # Skip very short/empty recordings
        continue
    
    # Feature Engineering
    features = []
    # For each sensor axis (x, y, z for accel and gyro)
    for sensor_type in ['/accelerometer', '/gyroscope']:
        sensor_data = group[group['sensor'] == sensor_type]
        if not sensor_data.empty:
            for axis in ['x', 'y', 'z']:
                features.extend([
                    sensor_data[axis].mean(),
                    sensor_data[axis].std(),
                    sensor_data[axis].min(),
                    sensor_data[axis].max(),
                    np.abs(sensor_data[axis]).mean() # Mean of absolute values
                ])
        else:
            # If a sensor is missing for a recording, pad with zeros
            features.extend([0] * 5 * 3)
            
    X.append(features)
    y.append(gesture_name)

print(f"Processed into {len(X)} feature sets.")

# --- Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling is crucial for SVC
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM Classifier
model = SVC(kernel='rbf', probability=True, gamma='auto') # RBF kernel is a good default
model.fit(X_train_scaled, y_train)

# --- Evaluate and Save ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

if accuracy < 0.85:
    print("WARNING: Accuracy is low. Consider recording more or better data.")
else:
    print("Model training successful!")

# Save the trained model and the scaler
joblib.dump(model, 'conductor_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved to disk.")