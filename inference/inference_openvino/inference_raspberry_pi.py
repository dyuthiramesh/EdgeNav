import cv2
import numpy as np
import openvino as ov
import time
import threading
import subprocess
import os
from openvino.preprocess import PrePostProcessor
from picamera2 import Picamera2

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEPTH_MODEL_XML = "best_openvino_depth_model/depth_model.xml"
OBJ_MODEL_XML   = "best_openvino_object_model/best.xml"

# Piper TTS Paths
HOME_DIR = os.path.expanduser("~")
PIPER_EXEC = f"{HOME_DIR}/piper_tts/piper/piper"
PIPER_MODEL = f"{HOME_DIR}/piper_tts/en_US-lessac-low.onnx"

CLASSES = ['bed', 'sofa', 'chair', 'table', 'lamp', 'tv', 'laptop', 'wardrobe', 'window', 'door', 'potted plant', 'photo frame', 'person']

# Timing & Thresholds
SPEECH_COOLDOWN = 5      # Reduced slightly for better responsiveness
STABILITY_THRESHOLD = 6  # Higher frames for better accuracy
GLOBAL_VOICE_DELAY = 1.5  # Minimum gap between any two different announcements

# Global State
last_spoken_time = {}
last_global_speech = 0
presence_counters = {cls: 0 for cls in CLASSES}
last_position = {cls: "" for cls in CLASSES}

# ==========================================
# 2. NON-BLOCKING VOICE ENGINE
# ==========================================
def speak(text):
    """Handles speech in a separate thread using Piper TTS to prevent UI lag."""
    global last_global_speech
    
    # Global cooldown to prevent jumbled words
    if (time.time() - last_global_speech) < GLOBAL_VOICE_DELAY:
        return

    def run_speech():
        # Clean text to prevent bash injection issues with quotes
        safe_text = text.replace("'", "")
        
        # Echo the text into Piper, output WAV to stdout (-), and pipe into aplay
        # 2>/dev/null suppresses Piper's verbose loading logs
        cmd = f"echo '{safe_text}' | {PIPER_EXEC} --model {PIPER_MODEL} --output_file - 2>/dev/null | aplay -q"
        
        subprocess.run(cmd, shell=True)
    
    last_global_speech = time.time()
    threading.Thread(target=run_speech, daemon=True).start()

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read()) / 1000.0
    except: return 0.0

# ==========================================
# 3. INITIALIZE MODELS
# ==========================================
core = ov.Core()
compiled_depth = core.compile_model(core.read_model(DEPTH_MODEL_XML), "AUTO")
depth_request = compiled_depth.create_infer_request()

ov_obj_model = core.read_model(OBJ_MODEL_XML)
ppp = PrePostProcessor(ov_obj_model)
ppp.input().tensor().set_layout(ov.Layout('NHWC')).set_element_type(ov.Type.u8)
ppp.input().model().set_layout(ov.Layout('NCHW'))
ppp.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
compiled_obj = core.compile_model(ppp.build(), "AUTO")
obj_request = compiled_obj.create_infer_request()

# ==========================================
# 4. CAMERA
# ==========================================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)

# ==========================================
# 5. MAIN LOOP
# ==========================================
prev_time = 0
frame_count = 0
current_temp = get_cpu_temp()

try:
    while True:
        frame_count += 1
        rgb_frame = picam2.capture_array()
        
        # --- FIX: ROTATE 180 (Upside Down) ---
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, -1) 
        H, W = frame.shape[:2]

        # --- DEPTH ESTIMATION ---
        depth_inp = cv2.resize(frame, (256, 256))
        depth_inp = cv2.cvtColor(depth_inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth_inp = (depth_inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        depth_inp = depth_inp.transpose(2, 0, 1)[np.newaxis]

        depth_results = depth_request.infer({0: depth_inp})[compiled_depth.output(0)]
        pred_depth = np.squeeze(depth_results)
        depth_norm = cv2.normalize(pred_depth, None, 0, 1, cv2.NORM_MINMAX)

        # --- OBJECT DETECTION ---
        obj_input = cv2.resize(frame, (416, 416))
        obj_results = obj_request.infer({0: np.expand_dims(obj_input, 0)})[compiled_obj.output(0)]
        obj_results = np.squeeze(obj_results).T 
        
        boxes, scores, class_ids = [], [], []
        detected_this_frame = {}

        for row in obj_results:
            score = np.max(row[4:])
            if score > 0.55: # Confidence threshold
                cx, cy, w, h = row[:4]
                x1 = int((cx - w/2) * W / 416)
                y1 = int((cy - h/2) * H / 416)
                boxes.append([x1, y1, int(w * W / 416), int(h * H / 416)])
                scores.append(float(score))
                class_ids.append(np.argmax(row[4:]))

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.5)

        # --- LOGIC: DEPTH PER OBJECT ---
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[idx]
            label = CLASSES[class_ids[idx]]
            
            # 1. Spatial Positioning
            obj_cx = x + (w / 2)
            pos = "left" if obj_cx < (W/3) else "right" if obj_cx > (2*W/3) else "center"
            
            # 2. Depth specifically for THIS object's bounding box
            # We map the object box back to the depth map scale
            dx1, dy1 = int(x * 256 / W), int(y * 256 / H)
            dx2, dy2 = int((x+w) * 256 / W), int((y+h) * 256 / H)
            
            # Ensure ROI is within bounds
            obj_roi = depth_norm[max(0, dy1):min(256, dy2), max(0, dx1):min(256, dx2)]
            avg_d = np.median(obj_roi) if obj_roi.size > 0 else 0
            
            dist_str = "far"
            if avg_d > 0.75: dist_str = "very close"
            elif avg_d > 0.45: dist_str = "near"

            detected_this_frame[label] = (pos, dist_str)
            
            # Visuals
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {dist_str}", (x, y-10), 1, 0.8, (0, 255, 0), 1)

        # --- VOICE & COOLDOWN LOGIC ---
        curr_now = time.time()
        for cls in CLASSES:
            if cls in detected_this_frame:
                presence_counters[cls] += 1
                pos, dist = detected_this_frame[cls]
                
                # Condition: Stable detection + (Cooldown passed OR position/distance changed significantly)
                if presence_counters[cls] >= STABILITY_THRESHOLD:
                    time_diff = curr_now - last_spoken_time.get(cls, 0)
                    state_changed = (pos != last_position.get(cls, ""))
                    
                    if time_diff > SPEECH_COOLDOWN or state_changed:
                        speak(f"{cls} {dist} on {pos}")
                        last_spoken_time[cls] = curr_now
                        last_position[cls] = pos
            else:
                presence_counters[cls] = 0 # Reset counter if object is gone

        # --- TELEMETRY DISPLAY ---
        fps = 1 / (curr_now - prev_time) if prev_time > 0 else 0
        prev_time = curr_now
        if frame_count % 30 == 0: current_temp = get_cpu_temp()

        # Build HUD
        heatmap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heatmap = cv2.resize(heatmap, (W, H))
        combined = np.hstack((frame, heatmap))
        
        cv2.putText(combined, f"FPS: {int(fps)} | CPU: {current_temp:.1f}C", (10, 25), 1, 1, (0, 255, 255), 2)
        cv2.imshow("Vision Assistant", combined)
        
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt: pass
finally:
    picam2.stop()
    cv2.destroyAllWindows()