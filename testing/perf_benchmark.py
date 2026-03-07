import tensorflow as tf
import numpy as np
import time
import os

# CONFIGURATION
DEPTH_MODEL_PATH = 'midas_mobilenet_v1.keras'
YOLO_MODEL_PATH = 'ultimate_indoor_yolo.keras'
ITERATIONS = 100  # Number of frames to simulate
IMG_SIZE_DEPTH = 224
IMG_SIZE_YOLO = 416

def run_benchmark():
    print("--- Performance Testing Initialized ---")
    
    # 1. Load Models
    print("Loading models into memory...")
    start_load = time.time()
    depth_model = tf.keras.models.load_model(DEPTH_MODEL_PATH, compile=False)
    yolo_model = tf.keras.models.load_model(YOLO_MODEL_PATH, compile=False)
    print(f"Models loaded in {time.time() - start_load:.2f} seconds.")

    # 2. Prepare Dummy Data
    d_input = np.random.randn(1, IMG_SIZE_DEPTH, IMG_SIZE_DEPTH, 3).astype(np.float32)
    y_input = np.random.randn(1, IMG_SIZE_YOLO, IMG_SIZE_YOLO, 3).astype(np.float32)

    latencies = []

    print(f"Starting benchmark for {ITERATIONS} iterations...")
    
    for i in range(ITERATIONS):
        iter_start = time.perf_counter()

        # --- Simulated Pipeline ---
        # A. Depth Inference
        _ = depth_model(d_input, training=False)
        
        # B. YOLO Inference
        _ = yolo_model(y_input, training=False)
        
        # C. Integration Logic (Simplified math)
        _ = np.median(np.random.rand(50, 50)) 

        iter_end = time.perf_counter()
        latencies.append(iter_end - iter_start)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{ITERATIONS} frames...")

    # 3. Calculate Metrics
    avg_latency = np.mean(latencies)
    fps = 1.0 / avg_latency
    p99_latency = np.percentile(latencies, 99) * 1000 # Worst case in ms
    std_dev = np.std(latencies) * 1000

    print("\n" + "="*30)
    print("      PERFORMANCE RESULTS      ")
    print("="*30)
    print(f"Average FPS:         {fps:.2f}")
    print(f"Average Latency:     {avg_latency*1000:.2f} ms")
    print(f"P99 Latency (Max):   {p99_latency:.2f} ms")
    print(f"Latency Stability:   +/- {std_dev:.2f} ms")
    print("="*30)

    if fps < 5:
        print("WARNING: FPS is below the safety threshold (5 FPS) for real-time navigation!")
    else:
        print("SUCCESS: Performance meets the navigation requirements.")

if __name__ == "__main__":
    if os.path.exists(DEPTH_MODEL_PATH) and os.path.exists(YOLO_MODEL_PATH):
        run_benchmark()
    else:
        print("Error: .keras files not found in the current directory.")