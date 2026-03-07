import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ==========================================
# CONFIGURATION
# ==========================================
# MODEL_PATH = 'best_depth_edge_forced.keras'
# MODEL_PATH = 'best_depth_head_unfrozen.keras'
# MODEL_PATH = 'best_depth_rich_skips.keras'
# MODEL_PATH = 'best_depth_augmented.keras'
MODEL_PATH = 'midas_mobilenet_v1.keras'
# MODEL_PATH = 'best_depth_head.keras'
IMG_SIZE = 224

print("Loading model... (This takes a moment)")
# compile=False to avoid custom loss function issues during inference
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# Initialize the webcam (0 is usually the built-in laptop camera)
cap = cv2.VideoCapture(0)

# Set up the CLAHE contrast enhancer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # OpenCV captures in BGR format. 
    # Let's resize it first to save processing time.
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    
    # --- PREPROCESSING (CLAHE) ---
    # Convert BGR to LAB color space for contrast enhancement
    lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    
    # Merge back and convert to RGB (which the MobileNet backbone expects)
    rgb_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    # Prepare input tensor
    input_tensor = keras.applications.mobilenet_v3.preprocess_input(np.array(rgb_clahe, dtype=np.float32))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # --- INFERENCE ---
    # Predict the depth
    pred_depth = model.predict(input_tensor, verbose=0)[0, :, :, 0]
    
    # --- POST-PROCESSING (Guided Filter) ---
    # We use the resized RGB frame as the guide for the filter
    guide_image = rgb_clahe.astype(np.float32) / 255.0
    depth_to_filter = pred_depth.astype(np.float32)
    
    sharpened_depth = cv2.ximgproc.guidedFilter(
        guide=guide_image, 
        src=depth_to_filter, 
        radius=8, 
        eps=0.01, 
        dDepth=-1
    )
    
    # --- VISUALIZATION ---
    # Normalize the sharpened depth map to 0-255 so OpenCV can display it
    depth_normalized = cv2.normalize(sharpened_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply a colormap (PLASMA or INFERNO work great for depth)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
    
    # OpenCV expects BGR for display, so convert the RGB guide back to BGR for the side-by-side
    display_rgb = cv2.cvtColor(rgb_clahe, cv2.COLOR_RGB2BGR)
    
    # Stack the original frame and the depth map side-by-side
    combined_view = np.hstack((display_rgb, depth_colormap))
    
    # Enlarge the combined view so it's easier to see on your screen
    combined_view_large = cv2.resize(combined_view, (IMG_SIZE * 4, IMG_SIZE * 2))
    
    # Show the video window
    cv2.imshow('Live Depth Estimation', combined_view_large)
    
    # Press 'q' on your keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up hardware resources
cap.release()
cv2.destroyAllWindows()
