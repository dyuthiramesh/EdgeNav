import cv2
import numpy as np
import openvino as ov
import time
from openvino.preprocess import PrePostProcessor

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
DEPTH_MODEL_XML = "best_openvino_depth_model/depth_model.xml"
OBJ_MODEL_XML   = "best_openvino_object_model/best.xml"
INPUT_SIZE_DEPTH = 256
CLASSES = ['bed', 'sofa', 'chair', 'table', 'lamp', 'tv', 'laptop', 'wardrobe', 'window', 'door', 'potted plant', 'photo frame', 'person']

# ==========================================
# 2. HELPERS & DEPTH PREPROCESSING
# ==========================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# Define mean and std once globally to save CPU cycles inside the loop
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_depth(frame):
    img  = cv2.cvtColor(apply_clahe(frame), cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (INPUT_SIZE_DEPTH, INPUT_SIZE_DEPTH)).astype(np.float32) / 255.0
    img  = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 256, 256)

# ==========================================
# 3. INITIALIZE OPENVINO CORE & MODELS
# ==========================================
print("[INFO] Initializing OpenVINO Core...")
core = ov.Core()

# --- Load Depth Model ---
print(f"[INFO] Loading Depth Model: {DEPTH_MODEL_XML}")
ov_depth_model = core.read_model(DEPTH_MODEL_XML)
compiled_depth_model = core.compile_model(ov_depth_model, "AUTO")
depth_infer_request = compiled_depth_model.create_infer_request()

# --- Load Object Detection Model ---
print(f"[INFO] Loading Object Model: {OBJ_MODEL_XML}")
ov_obj_model = core.read_model(OBJ_MODEL_XML)

# Object Model PrePostProcessor Setup (Offloads work to OpenVINO)
ppp = PrePostProcessor(ov_obj_model)
ppp.input().tensor().set_layout(ov.Layout('NHWC')).set_element_type(ov.Type.u8)
ppp.input().model().set_layout(ov.Layout('NCHW'))
ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
ov_obj_model = ppp.build()

compiled_obj_model = core.compile_model(ov_obj_model, "AUTO")
obj_infer_request = compiled_obj_model.create_infer_request()

# ==========================================
# 4. WEBCAM INFERENCE LOOP
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

print("[INFO] System Ready. Press 'q' to quit.")

prev_time = 0
prev_depth = None
alpha = 0.7  # Temporal smoothing factor for depth

while True:
    ret, frame = cap.read()
    if not ret: break

    H, W = frame.shape[:2]

    # ----------------------------------------
    # STEP A: DEPTH ESTIMATION
    # ----------------------------------------
    depth_inp = preprocess_depth(frame)
    
    # Run Inference
    depth_results = depth_infer_request.infer({0: depth_inp})[compiled_depth_model.output(0)]
    pred_depth = np.squeeze(depth_results)  # (256, 256)

    # Temporal smoothing to reduce flicker
    if prev_depth is None: prev_depth = pred_depth
    pred_depth = alpha * pred_depth + (1 - alpha) * prev_depth
    prev_depth = pred_depth

    # Colorize — invert so bright=far, dark=near
    depth_vis = cv2.normalize(pred_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_inv = 255 - depth_vis
    heatmap   = cv2.applyColorMap(depth_inv, cv2.COLORMAP_INFERNO)
    heatmap   = cv2.resize(heatmap, (W, H))

    # ----------------------------------------
    # STEP B: OBJECT DETECTION
    # ----------------------------------------
    resized_frame = cv2.resize(frame, (416, 416))
    obj_input_tensor = np.expand_dims(resized_frame, 0)

    # Run Inference
    obj_results = obj_infer_request.infer({0: obj_input_tensor})[compiled_obj_model.output(0)]
    obj_results = np.squeeze(obj_results).T 
    
    boxes, scores, class_ids = [], [], []
    for row in obj_results:
        score = np.max(row[4:])
        if score > 0.45:
            cx, cy, w, h = row[:4]
            # Rescale boxes back to original frame size
            x1 = int((cx - w/2) * W / 416)
            y1 = int((cy - h/2) * H / 416)
            boxes.append([x1, y1, int(w * W / 416), int(h * H / 416)])
            scores.append(float(score))
            class_ids.append(np.argmax(row[4:]))

    # NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.5)

    for i in indices:
        # Flatten index to prevent crashes across different OpenCV versions
        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = boxes[idx]
        label = CLASSES[class_ids[idx]]
        
        # Draw bounding box and label on the ORIGINAL frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {scores[idx]:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ----------------------------------------
    # STEP C: DISPLAY & FPS
    # ----------------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # Stack the bounding box frame and the depth heatmap side-by-side
    combined_view = np.hstack((frame, heatmap))
    
    # Overlay FPS
    cv2.putText(combined_view, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Vision Assistant 2026: Unified OpenVINO", combined_view)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()