import os
import sys
import time
import random
import numpy as np

# Configuration
SAMPLE_RATE = 50          # 50 Hz sampling rate
WINDOW_DURATION = 2       # 2-second windows
WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION  # 100 samples per window
NUM_AXES = 3              # X, Y, Z axes
CLASSES = ["normal", "misalignment", "bearing_failure", "off"]
NUM_CLASSES = len(CLASSES)

# Feature extraction output size: 5 features per axis * 3 axes = 15 features
# Features: [Mean, RMS, Peak-to-Peak, Low-Freq Power, High-Freq Power]
FEATURE_DIM = 15
HIDDEN_DIM = 16
LEARNING_RATE = 0.05
EPOCHS = 120
BATCH_SIZE = 16

# Console color codes for premium visual output
COLOR_HEADER = "\033[95m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

def print_header(title):
    print(f"\n{COLOR_BOLD}{COLOR_HEADER}" + "=" * 68)
    print(f" {title.center(66)} ")
    print("=" * 68 + f"{COLOR_RESET}")

# --- 1. Synthetic Vibration Sensor Data Generator ---
def generate_vibration_window(state):
    """
    Generates a 2-second window (100 samples x 3 axes) of simulated accelerometer/vibration data.
    X-axis: Radial vibration 1
    Y-axis: Radial vibration 2
    Z-axis: Axial acceleration (under gravity ~9.81 m/s^2)
    """
    t = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
    
    if state == "off":
        # Motor is powered down: minimal background noise, Z-axis registers gravity
        data = np.zeros((WINDOW_SIZE, NUM_AXES))
        data[:, 2] = 9.81
        data += np.random.normal(0, 0.03, (WINDOW_SIZE, NUM_AXES)) # tiny sensor noise
        
    elif state == "normal":
        # Healthy running motor: smooth high-speed rotation (~30 Hz)
        # Low vibration amplitude, steady signal with gravity offset on Z
        data = np.zeros((WINDOW_SIZE, NUM_AXES))
        data[:, 0] = 0.8 * np.sin(2 * np.pi * 30 * t)
        data[:, 1] = 0.8 * np.cos(2 * np.pi * 30 * t)
        data[:, 2] = 9.81 + 0.4 * np.sin(2 * np.pi * 30 * t)
        data += np.random.normal(0, 0.25, (WINDOW_SIZE, NUM_AXES))
        
    elif state == "misalignment":
        # Shaft misalignment: low-frequency wobble/sway (~6-8 Hz) with higher amplitudes
        data = np.zeros((WINDOW_SIZE, NUM_AXES))
        data[:, 0] = 3.5 * np.sin(2 * np.pi * 7 * t)
        data[:, 1] = 2.8 * np.cos(2 * np.pi * 7 * t)
        data[:, 2] = 9.81 + 1.2 * np.sin(2 * np.pi * 7 * t)
        data += np.random.normal(0, 0.4, (WINDOW_SIZE, NUM_AXES))
        
    elif state == "bearing_failure":
        # Damaged bearing: severe, irregular, high-frequency rattling/chattering.
        # High amplitude noise on all axes + transient high-g impact shocks (spalls)
        data = np.random.normal(0, 5.5, (WINDOW_SIZE, NUM_AXES))
        data[:, 2] += 9.81
        
        # Inject transient high-g shocks randomly (simulating mechanical impacts)
        for i in range(WINDOW_SIZE):
            if random.random() < 0.15:  # 15% chance of impact spike per sample
                shock_dir = random.choice([-1, 1])
                shock_val = shock_dir * random.uniform(12.0, 20.0)
                axis = random.choice([0, 1, 2])
                data[i, axis] += shock_val
                
    else:
        data = np.random.normal(0, 0.1, (WINDOW_SIZE, NUM_AXES))
        
    return data

def generate_dataset(samples_per_class=45):
    X_raw = []
    y = []
    
    for label_idx, state in enumerate(CLASSES):
        for _ in range(samples_per_class):
            window = generate_vibration_window(state)
            X_raw.append(window)
            y.append(label_idx)
            
    return X_raw, np.array(y)

# --- 2. DSP Block: Spectral Analysis Feature Extraction ---
def extract_spectral_features(window):
    """
    Simulates Edge Impulse's Spectral Analysis block.
    Extracts 5 features per axis:
      1. Mean: Capture orientation/gravity
      2. RMS: Signal energy
      3. Peak-to-Peak: Peak acceleration range (crucial for shock detection)
      4. Low-Freq Power (0 - 10 Hz): Wobble energy
      5. High-Freq Power (10 - 25 Hz): Bearing chatter energy
    """
    features = []
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0/SAMPLE_RATE)
    
    for axis in range(NUM_AXES):
        signal = window[:, axis]
        
        # 1. Statistical features
        mean_val = np.mean(signal)
        rms_val = np.sqrt(np.mean(signal**2))
        p2p_val = np.max(signal) - np.min(signal)
        
        # 2. FFT Spectral analysis
        # Detrend signal (subtract mean) for frequency domain analysis
        detrended = signal - mean_val
        fft_mag = np.abs(np.fft.rfft(detrended))
        
        # Split power into low-freq and high-freq bands
        low_band_mask = (freqs <= 10.0)
        high_band_mask = (freqs > 10.0)
        
        # Sum of squared magnitudes (spectral energy)
        low_power = np.sum(fft_mag[low_band_mask] ** 2)
        high_power = np.sum(fft_mag[high_band_mask] ** 2)
        
        # Convert to logarithmic scale (dB-like) for scaling stability
        log_low_power = np.log10(low_power + 1e-5)
        log_high_power = np.log10(high_power + 1e-5)
        
        features.extend([mean_val, rms_val, p2p_val, log_low_power, log_high_power])
        
    return np.array(features, dtype=np.float32)

def preprocess_dataset(X_raw):
    """
    Applies the DSP block to all raw windows and scales the features.
    """
    X_feats = []
    for window in X_raw:
        X_feats.append(extract_spectral_features(window))
    X_feats = np.array(X_feats)
    
    # Calculate normalization parameters (Mean and Standard Deviation scaling)
    mean = np.mean(X_feats, axis=0)
    std = np.std(X_feats, axis=0) + 1e-8
    
    scaled_feats = (X_feats - mean) / std
    return scaled_feats, mean, std

# --- 3. Neural Network Classifier Block (NumPy Multi-Layer Perceptron) ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    # Stabilized Softmax to avoid overflow
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetworkClassifier:
    """
    Standard MLP Neural Network trained locally on the edge features.
    """
    def __init__(self):
        # He (Kaiming) Weight Initialization
        self.W1 = np.random.randn(FEATURE_DIM, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / FEATURE_DIM)
        self.b1 = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
        self.W2 = np.random.randn(HIDDEN_DIM, NUM_CLASSES).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
        self.b2 = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        
        # Momentum parameters
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.momentum = 0.9

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def train_step(self, X, y_onehot):
        batch_size = X.shape[0]
        
        # 1. Forward propagation
        predictions = self.forward(X)
        
        # 2. Backward propagation (gradients)
        dZ2 = predictions - y_onehot
        dW2 = np.dot(self.A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        # 3. Update parameters with SGD + Momentum
        self.vW1 = self.momentum * self.vW1 - LEARNING_RATE * dW1
        self.vb1 = self.momentum * self.vb1 - LEARNING_RATE * db1
        self.vW2 = self.momentum * self.vW2 - LEARNING_RATE * dW2
        self.vb2 = self.momentum * self.vb2 - LEARNING_RATE * db2
        
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2
        
        # Cross-Entropy Loss computation
        loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-15), axis=1))
        return loss

def to_one_hot(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# --- 4. UI Dashboard: Live Sensor Telemetry Plots ---
def draw_ascii_waveforms(window):
    """
    Draws a visual dashboard of the live 3-axis accelerometer sensor streams.
    Displays the first 30 samples of the window (showing frequency and amplitude).
    """
    print(f"\n{COLOR_CYAN}  --- Live Accelerometer Real-Time Telemetry (First 30 Samples) ---{COLOR_RESET}")
    print("  Time   | X-Axis (Radial 1)  | Y-Axis (Radial 2)  | Z-Axis (Axial + Gravity)")
    print("  -------+--------------------+--------------------+-------------------------")
    
    for i in range(30):
        t_ms = i * 20 # 50Hz sample rate = 20ms intervals
        x = window[i, 0]
        y = window[i, 1]
        z = window[i, 2]
        
        # Helper to generate ASCII bar for visualization
        def make_bar(val, scale=1.2, is_z=False):
            # Centered around zero (or 9.8 for Z gravity)
            offset = 9.81 if is_z else 0.0
            adjusted = (val - offset) * scale
            num_ticks = int(round(abs(adjusted)))
            num_ticks = min(num_ticks, 9)
            
            if num_ticks == 0:
                bar = "."
            elif adjusted > 0:
                bar = "+" * num_ticks
            else:
                bar = "-" * num_ticks
                
            # Align in a 10 character space
            if adjusted >= 0:
                return f"|{bar}".ljust(10)
            else:
                return f"{bar}|".rjust(10)

        x_bar = make_bar(x, scale=1.5)
        y_bar = make_bar(y, scale=1.5)
        z_bar = make_bar(z, scale=1.5, is_z=True)
        
        print(f"  {t_ms:3d} ms | {x_bar} ({x:5.1f}) | {y_bar} ({y:5.1f}) | {z_bar} ({z:5.1f})")

def print_edge_suitability_summary(state, inference_time_ms):
    """
    Explains the edge computing business case based on the current detected state.
    """
    print(f"\n{COLOR_BOLD}{COLOR_YELLOW}  💡 EDGE COMPUTING BUSINESS CASE JUSTIFICATION FOR THIS INFERENCE:{COLOR_RESET}")
    
    # 1. Bandwidth calculation
    # 100 samples * 3 axes * 4 bytes per float = 1,200 bytes per 2s window.
    # Across a plant with 10,000 sensors running 24/7:
    # 1200 bytes * 0.5 windows/sec * 10,000 machines = 6 MB/sec = 518 GB/day of raw telemetry.
    # Edge reduces this to just state messages (4 classes = 1 byte).
    # 1 byte * 0.5 * 10000 = 5 KB/sec = 432 MB/day (a 99.9% savings).
    
    print(f"  • {COLOR_BOLD}Bandwidth:{COLOR_RESET} Local DSP reduced {COLOR_CYAN}300 raw vibration floats{COLOR_RESET} into {COLOR_CYAN}15 features{COLOR_RESET}.")
    print("    Instead of sending raw sensor streams, only the 1-byte classified state is transmitted.")
    
    # 2. Latency calculation
    print(f"  • {COLOR_BOLD}Latency:{COLOR_RESET} The edge microcontroller classified the state in {COLOR_GREEN}{inference_time_ms:.3f} ms{COLOR_RESET}.")
    
    if state == "bearing_failure":
        print(f"    🚨 {COLOR_RED}{COLOR_BOLD}CRITICAL SYSTEM DETECTED!{COLOR_RESET} Local control loops can immediately cut power")
        print("    to the motor in < 2 ms. A cloud-dependent system would add 200ms - 3s of latency,")
        print("    resulting in complete motor seizure, secondary gear damage, and safety hazards.")
    elif state == "misalignment":
        print(f"    ⚠️ {COLOR_YELLOW}WARNING:{COLOR_RESET} Machine wobble detected. Logged locally. Avoids spamming the factory network.")
        print("    Alert queued for the next scheduled maintenance cycle.")
    elif state == "normal":
        print("    ✅ Operating healthy. No action required. 100% of data processed locally, 0 bytes sent to cloud.")
    elif state == "off":
        print("    💤 Machine powered down. Sensor goes into low-power sleep mode, waking only on movement.")
        
    print(f"  • {COLOR_BOLD}Privacy & Uptime:{COLOR_RESET} Processed entirely offline. If the factory internet goes down,")
    print("    machine protection remains 100% active, preventing unmonitored failures.")

def main():
    print_header("EDGE IMPULSE SIMULATOR: INDUSTRIAL MOTOR HEALTH PREDICTIVE MAINTENANCE")
    print(" This simulator demonstrates processing high-frequency physical telemetry")
    print(" (3-Axis Accelerometer) locally on a low-power edge microcontroller.")
    print("-" * 68)
    
    # 1. Data Generation
    print(f"\n[Step 1/5] {COLOR_BOLD}Data Acquisition:{COLOR_RESET} Simulating sensor collection on the edge...")
    raw_windows, y = generate_dataset(45)
    print(f"  * Generated {len(raw_windows)} time-series windows (50Hz sample rate, 2-sec duration)")
    print("  * Balanced training data per class:")
    for idx, state in enumerate(CLASSES):
        print(f"    - {state.upper():16}: {np.sum(y == idx)} windows")
    
    # Shuffle and Train/Test Split
    indices = np.arange(y.size)
    np.random.seed(42)
    np.random.shuffle(indices)
    raw_windows = [raw_windows[i] for i in indices]
    y = y[indices]
    
    split_idx = int(0.8 * len(raw_windows))
    raw_train, raw_test = raw_windows[:split_idx], raw_windows[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    time.sleep(0.8)

    # 2. Signal Processing (DSP)
    print(f"\n[Step 2/5] {COLOR_BOLD}Signal Processing (DSP Block):{COLOR_RESET} Spectral Analysis...")
    print("  * Extracted features per axis: Mean, RMS, Peak-to-Peak, Log Low-Freq & High-Freq Power.")
    print("  * Replicated Edge Impulse's FFT band-power block to separate wobble from chatter.")
    print("  * Flattened 300 raw readings (100 steps * 3 axes) -> 15 scaled features.")
    
    X_train, dsp_mean, dsp_std = preprocess_dataset(raw_train)
    
    # Apply identical scaling parameters (stored in device ROM) to the test set
    X_test = []
    for w in raw_test:
        raw_feats = extract_spectral_features(w)
        scaled_feats = (raw_feats - dsp_mean) / dsp_std
        X_test.append(scaled_feats)
    X_test = np.array(X_test)
    
    y_train_onehot = to_one_hot(y_train, NUM_CLASSES)
    time.sleep(1.0)

    # 3. Model Training (Simulating Edge Impulse Studio)
    print(f"\n[Step 3/5] {COLOR_BOLD}Neural Network Training:{COLOR_RESET} Compiling 15x16x4 MLP classifier...")
    model = NeuralNetworkClassifier()
    
    num_batches = X_train.shape[0] // BATCH_SIZE
    
    for epoch in range(1, EPOCHS + 1):
        # Mini-batch gradient descent shuffle
        epoch_indices = np.arange(X_train.shape[0])
        np.random.shuffle(epoch_indices)
        X_shuffled = X_train[epoch_indices]
        y_shuffled_onehot = y_train_onehot[epoch_indices]
        
        epoch_loss = 0
        for b in range(num_batches):
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE
            batch_loss = model.train_step(X_shuffled[start:end], y_shuffled_onehot[start:end])
            epoch_loss += batch_loss
            
        epoch_loss /= num_batches
        
        # Calculate training accuracy
        train_preds = model.forward(X_train)
        train_acc = np.mean(np.argmax(train_preds, axis=1) == y_train)
        
        # Print progress every 15 epochs
        if epoch == 1 or epoch % 15 == 0:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Training Accuracy: {train_acc*100:6.2f}%")
            time.sleep(0.08) # Replicate training simulation feel
            
    print(f"  {COLOR_GREEN}✓ Training complete. NN weights and biases optimized.{COLOR_RESET}")
    time.sleep(0.5)

    # 4. Model Testing & Verification
    print(f"\n[Step 4/5] {COLOR_BOLD}Model Testing & Offline Validation:{COLOR_RESET}")
    test_preds = model.forward(X_test)
    test_predictions = np.argmax(test_preds, axis=1)
    test_acc = np.mean(test_predictions == y_test)
    print(f"  * Model Test Accuracy on unseen validation windows: {COLOR_BOLD}{COLOR_GREEN}{test_acc * 100:.2f}%{COLOR_RESET}")
    
    # Build Confusion Matrix
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_label, pred_label in zip(y_test, test_predictions):
        confusion_matrix[true_label, pred_label] += 1
        
    print(f"\n  {COLOR_BOLD}--- Confusion Matrix ---{COLOR_RESET}")
    print("  True \\ Pred      |  Normal  Misaligned  BearingFail  Off")
    print("  -----------------+----------------------------------------")
    for i, category in enumerate(CLASSES):
        print(f"  {category.upper():16} |   {confusion_matrix[i, 0]:2d}         {confusion_matrix[i, 1]:2d}          {confusion_matrix[i, 2]:2d}       {confusion_matrix[i, 3]:2d}")
    print("  -----------------+----------------------------------------")
    time.sleep(1.2)

    # 5. Interactive Live Edge Inference
    print_header("LIVE EDGE IMPULSE SIMULATOR SANDBOX")
    print(" The optimized code and DSP pipelines are compiled into C++ equivalent")
    print(" instructions and running locally in your machine's simulated memory.")
    print(" Select a physical motor condition to simulate and run edge inference:")
    
    while True:
        print(f"\n{COLOR_BOLD}Select Motor Condition to Simulate:{COLOR_RESET}")
        print(f" [{COLOR_GREEN}1{COLOR_RESET}] Healthy Motor (Normal rotation)")
        print(f" [{COLOR_YELLOW}2{COLOR_RESET}] Structural Misalignment (Low-frequency wobble)")
        print(f" [{COLOR_RED}3{COLOR_RESET}] Bearing Failure (Severe vibration + transient impact shocks)")
        print(f" [{COLOR_BLUE}4{COLOR_RESET}] Motor Shut Down (Inactive state)")
        print(f" [{COLOR_CYAN}5{COLOR_RESET}] Read Edge Computing Business Case Justification")
        print(" [6] Exit Simulator")
        
        choice = input("Enter choice [1-6]: ").strip()
        
        if choice == '6':
            print(f"\n{COLOR_BOLD}Exiting. Industrial predictive maintenance simulator shut down successfully!{COLOR_RESET}")
            break
        elif choice == '5':
            print_header("EDGE COMPUTING BUSINESS JUSTIFICATION SUMMARY")
            print(f"1. {COLOR_BOLD}Bandwidth Constraints:{COLOR_RESET}")
            print("   Vibration data is high-frequency (50Hz - 5kHz). Streaming continuous raw data")
            print("   creates network congestion and huge cloud billing. Edge DSP reduces this by >99%.")
            print(f"2. {COLOR_BOLD}Response Latency:{COLOR_RESET}")
            print("   Severe mechanical malfunctions can cause explosive physical failure. Local edge")
            print("   inference detects failures in <2ms, allowing instant emergency shutoffs.")
            print(f"3. {COLOR_BOLD}Offline Independence (Uptime):{COLOR_RESET}")
            print("   Industrial settings have spotty WiFi/cellular connections. Offline edge compute")
            print("   guarantees that critical safety monitoring never stops due to connection drops.")
            print(f"4. {COLOR_BOLD}Security & Data Privacy:{COLOR_RESET}")
            print("   Vibration patterns reveal plant capacity, shift schedules, and machinery specs.")
            print("   Keeping raw telemetry on-device prevents industrial espionage.")
            print("=" * 68)
            input("\nPress Enter to return to the menu...")
            continue
        elif choice not in ['1', '2', '3', '4']:
            print(f"{COLOR_RED}Invalid input. Please choose a number from 1 to 6.{COLOR_RESET}")
            continue
            
        selected_state = CLASSES[int(choice) - 1]
        print(f"\nGenerating live 50Hz sensor stream for: {COLOR_BOLD}{selected_state.upper()}{COLOR_RESET}")
        time.sleep(0.3)
        
        live_window = generate_vibration_window(selected_state)
        
        # Display visual ASCII graph of the waves
        draw_ascii_waveforms(live_window)
        print("-" * 68)
        
        # Local DSP & Inference
        t_start = time.perf_counter()
        
        # 1. Run local DSP Spectral Analysis block
        raw_features = extract_spectral_features(live_window)
        # 2. Scale features using parameters stored in memory
        scaled_features = (raw_features - dsp_mean) / dsp_std
        # 3. Feed to local Neural Network
        prediction_probs = model.forward(scaled_features.reshape(1, -1))[0]
        predicted_idx = np.argmax(prediction_probs)
        
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000
        
        # Output results
        print(f"  * {COLOR_BOLD}Local Processing Time (DSP + Neural Net):{COLOR_RESET} {COLOR_GREEN}{inference_time_ms:.4f} ms{COLOR_RESET}")
        print("  * Classification Results:")
        for idx, category in enumerate(CLASSES):
            prob = prediction_probs[idx]
            filled_chars = int(round(prob * 16))
            bar_color = COLOR_GREEN if idx == predicted_idx else COLOR_BLUE
            
            # Simple custom bar representation
            bar = "[" + bar_color + "█" * filled_chars + COLOR_RESET + "░" * (16 - filled_chars) + "]"
            marker = f"{COLOR_BOLD}◀ DETECTED{COLOR_RESET}" if idx == predicted_idx else ""
            
            # Text formatting
            print(f"    - {category.capitalize():16}: {bar} {prob*100:6.2f}% {marker}")
            
        print("-" * 68)
        
        # Print suitability description
        print_edge_suitability_summary(CLASSES[predicted_idx], inference_time_ms)
        print("=" * 68)
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
