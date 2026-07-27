import os
import sys
import time
import random
import numpy as np

# Configuration
SAMPLE_RATE = 50          # 50 Hz (50 samples per second)
WINDOW_DURATION = 2       # 2-second windows
WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION # 100 samples per window
NUM_AXES = 3              # X, Y, Z accelerometer axes
CLASSES = ["idle", "wave", "shake"]
NUM_CLASSES = len(CLASSES)

# Feature extraction output size: 4 features per axis (mean, std, rms, max) * 3 axes = 12 features
FEATURE_DIM = 12
HIDDEN_DIM = 16
LEARNING_RATE = 0.1
EPOCHS = 100

def print_header(title):
    print("=" * 65)
    print(f" {title.center(63)} ")
    print("=" * 65)

# --- 1. Synthetic Sensor Data Generator ---
def generate_sensor_window(gesture_type):
    """
    Generates a 2-second window (100 samples x 3 axes) of simulated accelerometer data.
    X-axis: Index 0, Y-axis: Index 1, Z-axis: Index 2.
    Gravity is assumed to be on the Z-axis (~9.8 m/s^2).
    """
    t = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
    noise = np.random.normal(0, 0.2, (WINDOW_SIZE, NUM_AXES))
    
    if gesture_type == "idle":
        # Sitting flat: gravity on Z, minimal noise
        data = np.zeros((WINDOW_SIZE, NUM_AXES))
        data[:, 2] = 9.81
        data += np.random.normal(0, 0.05, (WINDOW_SIZE, NUM_AXES))
        
    elif gesture_type == "wave":
        # Waving left/right: slow large sine wave on X-axis, smaller waves on Y
        data = np.zeros((WINDOW_SIZE, NUM_AXES))
        data[:, 0] = 5.0 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz swing
        data[:, 1] = 1.5 * np.cos(2 * np.pi * 1.5 * t)
        data[:, 2] = 9.81 + np.random.normal(0, 0.2, WINDOW_SIZE)
        data += noise
        
    elif gesture_type == "shake":
        # Shaking vigorously: high frequency, high amplitude random noise on all axes
        data = np.random.normal(0, 8.0, (WINDOW_SIZE, NUM_AXES))
        data[:, 2] += 9.81
        
    else:
        data = noise
        
    return data

def generate_dataset(samples_per_class=40):
    X_raw = []
    y = []
    
    for label_idx, gesture in enumerate(CLASSES):
        for _ in range(samples_per_class):
            window = generate_sensor_window(gesture)
            X_raw.append(window)
            y.append(label_idx)
            
    return X_raw, np.array(y)

# --- 2. Signal Processing (DSP Block) ---
def extract_spectral_features(window):
    """
    Simulates Edge Impulse's Spectral Analysis block.
    Extracts statistical features from the time-series window.
    Features: [mean_x, std_x, rms_x, max_x, mean_y, std_y, rms_y, max_y, mean_z, std_z, rms_z, max_z]
    """
    features = []
    for axis in range(NUM_AXES):
        signal = window[:, axis]
        mean = np.mean(signal)
        std = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        max_val = np.max(np.abs(signal))
        features.extend([mean, std, rms, max_val])
    return np.array(features)

def preprocess_dataset(X_raw):
    X_features = []
    for window in X_raw:
        features = extract_spectral_features(window)
        X_features.append(features)
    
    X_features = np.array(X_features)
    # Simple feature scaling/normalization (mean=0, std=1)
    mean = np.mean(X_features, axis=0)
    std = np.std(X_features, axis=0) + 1e-8
    scaled_features = (X_features - mean) / std
    return scaled_features, mean, std

# --- 3. Neural Network Classifier Block (NumPy) ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SensorMLP:
    """
    Neural Network classifier to classify movements locally.
    """
    def __init__(self):
        # He initialization
        self.W1 = np.random.randn(FEATURE_DIM, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / FEATURE_DIM)
        self.b1 = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
        self.W2 = np.random.randn(HIDDEN_DIM, NUM_CLASSES).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
        self.b2 = np.zeros((1, NUM_CLASSES), dtype=np.float32)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def train_step(self, X, y_onehot):
        batch_size = X.shape[0]
        
        # Forward Pass
        predictions = self.forward(X)
        
        # Backward Pass
        dZ2 = predictions - y_onehot
        dW2 = np.dot(self.A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        # Gradient Descent Updates
        self.W1 -= LEARNING_RATE * dW1
        self.b1 -= LEARNING_RATE * db1
        self.W2 -= LEARNING_RATE * dW2
        self.b2 -= LEARNING_RATE * db2

        loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-15), axis=1))
        return loss

def to_one_hot(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def draw_console_graph(window):
    """
    Draws a visual ASCII representation of a 3-axis accelerometer sensor stream
    to demonstrate live data processing.
    """
    print("  Live Accelerometer Signals (First 25 steps):")
    print("  Time | X-Axis (Wave)     | Y-Axis            | Z-Axis (Gravity)")
    print("  -----+-------------------+-------------------+-----------------")
    
    # We display 25 samples out of the 100 in the window to fit the console
    for i in range(0, 50, 2):
        x_val = window[i, 0]
        y_val = window[i, 1]
        z_val = window[i, 2]
        
        # Create small text graphs
        x_bar = ("#" * int(abs(x_val))).ljust(15) if x_val >= 0 else ("-" * int(abs(x_val))).rjust(15)
        y_bar = ("#" * int(abs(y_val))).ljust(15) if y_val >= 0 else ("-" * int(abs(y_val))).rjust(15)
        z_bar = ("#" * int(abs(z_val))).ljust(15)
        
        print(f"  {i*20:3d}ms | {x_bar} | {y_bar} | {z_bar}")

def main():
    print_header("EDGE IMPULSE LOCAL SENSOR SIMULATOR")
    print(" This simulator demonstrates processing high-frequency sensor data")
    print(" (3-Axis Accelerometer) locally on an edge device.")
    print("-" * 65)
    
    # 1. Data Generation
    print("\n[Step 1/4] Data Acquisition: Generating Simulated Accel Signals...")
    X_raw, y = generate_dataset(40)
    print(f"  * Collected {len(X_raw)} sensor windows (50Hz sample rate, 2-sec duration)")
    print(f"  * Classes: idle ({np.sum(y==0)}), wave ({np.sum(y==1)}), shake ({np.sum(y==2)})")
    time.sleep(1)

    # Shuffle and Train/Test Split
    indices = np.arange(y.size)
    np.random.seed(42)
    np.random.shuffle(indices)
    X_raw = [X_raw[i] for i in indices]
    y = y[indices]
    
    split = int(0.8 * len(X_raw))
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Signal Processing (DSP)
    print("\n[Step 2/4] Signal Processing: Local Spectral Analysis Block")
    print("  * Extracted features: Mean, Std Dev, RMS, and Max Amplitude per axis.")
    print("  * Reduced 300 raw measurements (100 steps x 3 axes) -> 12 key DSP features.")
    time.sleep(1)
    
    X_train_feats, dsp_mean, dsp_std = preprocess_dataset(X_train_raw)
    X_test_feats = (np.array([extract_spectral_features(w) for w in X_test_raw]) - dsp_mean) / dsp_std
    
    y_train_onehot = to_one_hot(y_train, NUM_CLASSES)

    # 3. Model Training
    print("\n[Step 3/4] Neural Network Training (Simulating Edge Impulse Studio)")
    model = SensorMLP()
    
    for epoch in range(1, EPOCHS + 1):
        loss = model.train_step(X_train_feats, y_train_onehot)
        train_preds = model.forward(X_train_feats)
        train_acc = np.mean(np.argmax(train_preds, axis=1) == y_train)
        
        if epoch == 1 or epoch % 20 == 0:
            print(f"  Epoch {epoch:03d}/{EPOCHS} -> Cross-Entropy Loss: {loss:.4f} | Training Acc: {train_acc*100:.1f}%")
            time.sleep(0.1)

    # Test Validation
    test_preds = model.forward(X_test_feats)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test)
    print(f"\nTraining complete! Local validation accuracy: {test_acc*100:.1f}%")
    time.sleep(1.5)

    # 4. Interactive Live Simulator
    print("\n[Step 4/4] Live Interactive Local Inference")
    print("The model is compiled and deployed locally to your device memory.")
    print("Select a motion to simulate and watch the device process it instantly:")
    
    while True:
        print("\nChoose simulated sensor action:")
        print(" [1] Simulate IDLE device (Sitting on desk)")
        print(" [2] Simulate WAVE gesture (Slow hand swing)")
        print(" [3] Simulate SHAKE gesture (High velocity shake)")
        print(" [4] Exit Simulator")
        
        choice = input("Enter choice [1-4]: ").strip()
        
        if choice == '4':
            print("\nExiting. Thank you for using the Edge Impulse local simulator!")
            break
        elif choice not in ['1', '2', '3']:
            print("Invalid input, please enter a number from 1 to 4.")
            continue
            
        selected_gesture = CLASSES[int(choice) - 1]
        print(f"\nGenerating live sensor data for: {selected_gesture.upper()}")
        live_window = generate_sensor_window(selected_gesture)
        time.sleep(0.5)
        
        # Display visual ASCII graph of the waves
        draw_console_graph(live_window)
        print("-" * 65)
        
        # Local DSP & Inference
        t_start = time.perf_counter()
        # 1. Run local DSP block
        raw_features = extract_spectral_features(live_window)
        # 2. Scale features using parameters stored in device memory
        scaled_features = (raw_features - dsp_mean) / dsp_std
        # 3. Feed to local Neural Network
        prediction_probs = model.forward(scaled_features.reshape(1, -1))[0]
        predicted_idx = np.argmax(prediction_probs)
        t_end = time.perf_counter()
        
        inference_time_ms = (t_end - t_start) * 1000
        
        # Output results
        print(f"  * Local Processing Time (DSP + Inference): {inference_time_ms:.3f} ms")
        print("  * Classification Results:")
        for idx, category in enumerate(CLASSES):
            prob = prediction_probs[idx]
            filled_chars = int(round(prob * 15))
            bar = "[" + "█" * filled_chars + "░" * (15 - filled_chars) + "]"
            marker = "<- DETECTED" if idx == predicted_idx else ""
            print(f"    - {category.capitalize():8}: {bar} {prob*100:6.2f}% {marker}")
            
        print("-" * 65)
        time.sleep(1)

if __name__ == "__main__":
    main()
