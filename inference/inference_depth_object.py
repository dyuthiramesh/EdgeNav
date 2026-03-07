# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import time

# # ==========================================
# # 1. CONFIGURATION & MODELS
# # ==========================================
# DEPTH_MODEL_PATH = 'midas_mobilenet_v1.keras'
# YOLO_MODEL_PATH = 'ultimate_indoor_yolo.keras'
# IMG_SIZE_DEPTH = 224
# IMG_SIZE_YOLO = 416

# # YOLO Specifics
# CONF_THRESHOLD = 0.15
# IOU_THRESHOLD = 0.2
# CLASSES = ["door", "cabinetDoor", "refrigeratorDoor", "window", "chair", 
#            "table", "cabinet", "couch", "openedDoor", "pole"]
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# print("Loading models... Please wait.")
# depth_model = tf.keras.models.load_model(DEPTH_MODEL_PATH, compile=False)
# yolo_model = tf.keras.models.load_model(YOLO_MODEL_PATH, compile=False)
# print("Models loaded successfully!")

# # ==========================================
# # 2. INITIALIZATION
# # ==========================================
# cap = cv2.VideoCapture(0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     if not ret: break

#     orig_h, orig_w = frame.shape[:2]
    
#     # ------------------------------------------
#     # 3. DEPTH ESTIMATION BRANCH
#     # ------------------------------------------
#     # Resize and CLAHE preprocessing
#     d_frame = cv2.resize(frame, (IMG_SIZE_DEPTH, IMG_SIZE_DEPTH))
#     lab = cv2.cvtColor(d_frame, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     cl = clahe.apply(l)
#     rgb_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
#     d_input = keras.applications.mobilenet_v3.preprocess_input(np.array(rgb_clahe, dtype=np.float32))
#     d_input = np.expand_dims(d_input, axis=0)
    
#     pred_depth = depth_model.predict(d_input, verbose=0)[0, :, :, 0]
    
#     # Post-processing (Guided Filter)
#     guide = rgb_clahe.astype(np.float32) / 255.0
#     sharpened = cv2.ximgproc.guidedFilter(guide, pred_depth.astype(np.float32), radius=8, eps=0.01, dDepth=-1)
    
#     # Depth Visualization
#     depth_norm = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
#     depth_color = cv2.resize(depth_color, (orig_w, orig_h)) # Resize back for display

#     # ------------------------------------------
#     # 4. OBJECT DETECTION BRANCH
#     # ------------------------------------------
#     y_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     y_frame = cv2.resize(y_frame, (IMG_SIZE_YOLO, IMG_SIZE_YOLO))
#     y_input = tf.expand_dims(y_frame / 255.0, 0)
    
#     y_preds = yolo_model.predict(y_input, verbose=0)
    
#     det_frame = frame.copy()
#     boxes, confs, class_ids = [], [], []

#     for pred in y_preds:
#         pred = pred[0]
#         grid = pred.shape[0]
#         pred = np.reshape(pred, (grid, grid, 3, 5 + len(CLASSES)))
        
#         # Decoding logic
#         box_xy = (1 / (1 + np.exp(-pred[..., 0:2]))) * [orig_w, orig_h]
#         box_wh = np.maximum(0, pred[..., 2:4]) * [orig_w, orig_h]
#         obj_score = 1 / (1 + np.exp(-pred[..., 4:5]))
#         cls_prob = 1 / (1 + np.exp(-pred[..., 5:]))
        
#         scores = obj_score * cls_prob
#         max_scores = np.max(scores, axis=-1)
#         mask = max_scores > CONF_THRESHOLD
        
#         box_mins = box_xy - (box_wh / 2.)
#         box_maxes = box_xy + (box_wh / 2.)
        
#         boxes.append(np.concatenate([box_mins, box_maxes], axis=-1)[mask])
#         confs.append(max_scores[mask])
#         class_ids.append(np.argmax(scores, axis=-1)[mask])

#     # NMS and Drawing
#     if any(len(b) > 0 for b in boxes):
#         boxes_flat = np.concatenate(boxes, axis=0)
#         confs_flat = np.concatenate(confs, axis=0)
#         ids_flat = np.concatenate(class_ids, axis=0)
        
#         indices = tf.image.non_max_suppression(boxes_flat, confs_flat, 15, IOU_THRESHOLD)
#         for i in indices.numpy():
#             b, s, c = boxes_flat[i], confs_flat[i], ids_flat[i]
#             x1, y1, x2, y2 = map(int, b)
#             color = COLORS[c]
#             cv2.rectangle(det_frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(det_frame, f"{CLASSES[c]} {s:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # ------------------------------------------
#     # 5. COMBINED DISPLAY
#     # ------------------------------------------
#     # Stack Detection and Depth side-by-side
#     combined = np.hstack((det_frame, depth_color))
    
#     fps = 1.0 / (time.time() - start_time)
#     cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    
#     cv2.imshow('AI Vision: Detection (Left) | Depth (Right)', cv2.resize(combined, (1280, 480)))
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEPTH_MODEL_PATH = 'midas_mobilenet_v1.keras'
YOLO_MODEL_PATH = 'ultimate_indoor_yolo.keras'
IMG_SIZE_DEPTH = 224
IMG_SIZE_YOLO = 416

# Calibration: Adjust this to match your specific webcam
# Increase this number if the meter readings are too low.
CALIBRATION_CONSTANT = 7.5 

CLASSES = ["door", "cabinetDoor", "refrigeratorDoor", "window", "chair", 
           "table", "cabinet", "couch", "openedDoor", "pole"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def estimate_meters(pixel_val):
    """
    Converts 'Ulta' depth values (0=Close, 255=Far) to meters.
    """
    # Inverse the 'ulta' value: Low pixel value now becomes high 'closeness'
    closeness_factor = 255.0 - pixel_val
    
    # Safeguard against zero
    closeness_factor = max(closeness_factor, 1.0)
    
    # Formula: meters = k / (normalized_closeness + epsilon)
    dist_m = CALIBRATION_CONSTANT / (closeness_factor / 255.0 + 0.05)
    return round(dist_m, 2)

# ==========================================
# 2. MODEL LOADING
# ==========================================
print("Loading Depth and Detection models...")
depth_model = tf.keras.models.load_model(DEPTH_MODEL_PATH, compile=False)
yolo_model = tf.keras.models.load_model(YOLO_MODEL_PATH, compile=False)
print("System Online!")

# ==========================================
# 3. INITIALIZE CAMERA & TOOLS
# ==========================================
cap = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    orig_h, orig_w = frame.shape[:2]

    # --- DEPTH PREPROCESSING ---
    d_frame = cv2.resize(frame, (IMG_SIZE_DEPTH, IMG_SIZE_DEPTH))
    lab = cv2.cvtColor(d_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    rgb_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    d_input = keras.applications.mobilenet_v3.preprocess_input(np.array(rgb_clahe, dtype=np.float32))
    d_input = np.expand_dims(d_input, axis=0)
    
    # --- DEPTH INFERENCE ---
    pred_depth = depth_model.predict(d_input, verbose=0)[0, :, :, 0]
    
    # Post-process for visualization
    depth_norm = cv2.normalize(pred_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Resize depth map to original frame size for coordinate matching
    depth_map_full = cv2.resize(depth_norm, (orig_w, orig_h))
    
    # Since visual 'close' is usually bright, we flip for the display colormap
    depth_display = cv2.applyColorMap(255 - depth_map_full, cv2.COLORMAP_PLASMA)

    # --- YOLO PREPROCESSING ---
    y_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    y_resized = cv2.resize(y_rgb, (IMG_SIZE_YOLO, IMG_SIZE_YOLO))
    y_input = tf.expand_dims(y_resized / 255.0, 0)
    
    # --- YOLO INFERENCE ---
    y_preds = yolo_model.predict(y_input, verbose=0)
    det_frame = frame.copy()

    # --- DECODE YOLO OUTPUTS ---
    for pred in y_preds:
        pred = pred[0]
        grid = pred.shape[0]
        pred = np.reshape(pred, (grid, grid, 3, 5 + len(CLASSES)))
        
        obj_score = 1 / (1 + np.exp(-pred[..., 4:5]))
        cls_prob = 1 / (1 + np.exp(-pred[..., 5:]))
        scores = obj_score * cls_prob
        
        mask = np.max(scores, axis=-1) > 0.15
        if not np.any(mask): continue

        box_xy = (1 / (1 + np.exp(-pred[..., 0:2]))) * [orig_w, orig_h]
        box_wh = np.maximum(0, pred[..., 2:4]) * [orig_w, orig_h]
        
        box_mins, box_maxes = box_xy - (box_wh / 2.), box_xy + (box_wh / 2.)
        boxes = np.concatenate([box_mins, box_maxes], axis=-1)[mask]
        confidences = np.max(scores, axis=-1)[mask]
        class_ids = np.argmax(scores, axis=-1)[mask]

        # NMS to filter overlapping boxes
        indices = tf.image.non_max_suppression(boxes, confidences, 10, 0.2)
        
        for i in indices.numpy():
            x1, y1, x2, y2 = map(int, boxes[i])
            # Clip coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            # --- GET DEPTH FOR OBJECT ---
            # Crop the depth map to the detected object's area
            roi_depth = depth_map_full[y1:y2, x1:x2]
            if roi_depth.size > 0:
                # Use median pixel value to calculate distance
                obj_pixel_val = np.median(roi_depth)
                meters = estimate_meters(obj_pixel_val)
            else:
                meters = 0.0

            color = COLORS[class_ids[i]]
            label = f"{CLASSES[class_ids[i]]}: {meters}m"
            
            # Drawing boxes and info
            cv2.rectangle(det_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(det_frame, (x1, y1-25), (x1+180, y1), color, -1)
            cv2.putText(det_frame, label, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # --- FINAL DISPLAY ---
    # Stack the detection view and depth view side-by-side
    output_view = np.hstack((det_frame, depth_display))
    
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(output_view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Spatial AI - Detection (L) | Depth (R)', cv2.resize(output_view, (1280, 480)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
