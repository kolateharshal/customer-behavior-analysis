import os
import sys
import numpy as np
from PIL import Image

# 1. Try to import TensorFlow Lite Interpreter
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Using TensorFlow's Interpreter")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using tflite_runtime's Interpreter")
    except ImportError:
        print("Error: Could not import either 'tensorflow' or 'tflite_runtime'.")
        print("Please install one of them to run this script:")
        print("  pip install tensorflow")
        print("  or")
        print("  pip install tflite-runtime")
        sys.exit(1)

def run_local_inference(model_path, image_path):
    """
    Loads a TensorFlow Lite model and runs classification on a target image.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please download your TensorFlow Lite model from Edge Impulse and place it here.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # 2. Load the TFLite model and allocate tensors
    print(f"Loading model: {model_path}...")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print input model details (to understand what shape the model expects)
    input_shape = input_details[0]['shape']
    print(f"Model expects input shape: {input_shape}") # e.g. [1, 96, 96, 3] for RGB 96x96
    expected_height = input_shape[1]
    expected_width = input_shape[2]

    # 3. Load and preprocess the image
    print(f"Loading image: {image_path}...")
    img = Image.open(image_path)
    
    # Edge Impulse models expect RGB images. Convert image to RGB if it isn't already.
    img = img.convert('RGB')
    
    # Resize the image to match the model's expected shape (e.g. 96x96)
    img_resized = img.resize((expected_width, expected_height), Image.Resampling.LANCZOS)
    
    # Convert image pixels to a numpy array
    img_array = np.array(img_resized, dtype=np.float32)

    # Edge Impulse normalizes pixel values. Usually images are normalized between 0 and 1.
    # We divide by 255.0 to scale pixel values from [0, 255] to [0.0, 1.0]
    img_array = img_array / 255.0

    # Add a batch dimension to match input shape [1, height, width, 3]
    input_data = np.expand_dims(img_array, axis=0)

    # If the model is Quantized (Int8), convert float32 values to int8 using the model's scale/zero-point
    # For now, we assume a standard Float32 model, which is easier to work with.
    # Note: Edge Impulse's Float32 model works directly with the normalized 0.0 - 1.0 float values.

    # 4. Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 5. Run inference
    print("Running inference locally...")
    interpreter.invoke()

    # 6. Extract the results
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    print("\n--- Inference Results ---")
    
    # Define labels in the exact alphabetical order they were trained in Edge Impulse:
    # Since we uploaded 'apple', 'banana', 'orange', the alphabetical order is:
    labels = ["apple", "banana", "orange"]
    
    for i, label in enumerate(labels):
        if i < len(output_data):
            confidence = output_data[i]
            # If the output is quantized, we might need to convert it (Float32 is already 0.0 - 1.0)
            print(f"{label}: {confidence * 100:.2f}%")

if __name__ == "__main__":
    # Example usage:
    # 1. Download your model from Edge Impulse (TFLite Float32 model).
    # 2. Rename it to 'model.lite' (or update the filename below).
    # 3. Choose one of your local dataset images to test.
    
    MODEL_FILE = "model.lite"
    TEST_IMAGE = "dataset/apple/apple_01.jpg"
    
    print("--- Edge Impulse Local Inference Script ---")
    print(f"Targeting Model file: {MODEL_FILE}")
    print(f"Targeting Test Image: {TEST_IMAGE}\n")
    
    run_local_inference(MODEL_FILE, TEST_IMAGE)
