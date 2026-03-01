import cv2
import numpy as np
import tensorflow as tf
import time

# 1. Configuration
MODEL_PATH = "ultimate_indoor_yolo.keras" # Change this to your saved model's name
CONF_THRESHOLD = 0.15 # Since you mentioned 0.15 is giving accurate boxes right now
IOU_THRESHOLD = 0.2

CLASSES = ["door", "cabinetDoor", "refrigeratorDoor", "window", "chair", 
           "table", "cabinet", "couch", "openedDoor", "pole"]

# Your custom K-Means anchors
ANCHORS = np.array([
    [[294, 98], [191, 353], [343, 333]],
    [[143, 67], [91, 221], [69, 344]],
    [[38, 33], [37, 86], [55, 148]]
])

# Distinct colors for each class (OpenCV uses BGR format)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 2. Load Model
print(f"Loading model {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# 3. Start Webcam
cap = cv2.VideoCapture(0) # '0' is usually the default laptop webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live inference... Press 'q' to quit.")

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
        
    original_h, original_w = frame.shape[:2]
    
    # Preprocess the frame for the model
    # OpenCV captures in BGR, but your model was trained on RGB images
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (416, 416))
    img_normalized = img_resized / 255.0
    input_tensor = tf.expand_dims(img_normalized, 0)
    
    # Run Inference
    predictions = model.predict(input_tensor, verbose=0)
    
    boxes, confidences, class_ids = [], [], []
    
    # Decode Tensors (Matching your successful logic)
    for i, pred in enumerate(predictions):
        pred = pred[0]
        grid_size = pred.shape[0]
        pred = np.reshape(pred, (grid_size, grid_size, 3, 5 + len(CLASSES)))
        
        box_xy = 1 / (1 + np.exp(-pred[..., 0:2])) 
        box_wh = np.maximum(0, pred[..., 2:4])     
        
        # Scale back to original webcam frame dimensions, NOT just 416
        box_xy = box_xy * [original_w, original_h]
        box_wh = box_wh * [original_w, original_h]
        
        objectness = 1 / (1 + np.exp(-pred[..., 4:5]))
        class_probs = 1 / (1 + np.exp(-pred[..., 5:])) 
        
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        abs_boxes = np.concatenate([box_mins, box_maxes], axis=-1)
        
        scores = objectness * class_probs
        max_scores = np.max(scores, axis=-1) 
            
        # --- ADD THIS LINE ---
        if np.max(max_scores) > 0.7:
            print(f"Grid {grid_size}x{grid_size} Max Score: {np.max(max_scores):.4f}")      
        max_class_ids = np.argmax(scores, axis=-1) 
        
        mask = max_scores > CONF_THRESHOLD 
        
        boxes.append(abs_boxes[mask])
        confidences.append(max_scores[mask]) 
        class_ids.append(max_class_ids[mask])

    if any(len(b) > 0 for b in boxes):
        boxes = np.concatenate(boxes, axis=0)
        confidences = np.concatenate(confidences, axis=0)
        class_ids = np.concatenate(class_ids, axis=0)

        # Apply NMS
        indices = tf.image.non_max_suppression(
            boxes, confidences, max_output_size=15, iou_threshold=IOU_THRESHOLD
        )
        
        final_boxes = boxes[indices.numpy()]
        final_confs = confidences[indices.numpy()]
        final_classes = class_ids[indices.numpy()]
        
        # Draw the boxes on the OpenCV frame
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            cls_idx = final_classes[i]
            score = final_confs[i]
            color = COLORS[cls_idx]
            
            xmin = max(0, int(box[0]))
            ymin = max(0, int(box[1]))
            xmax = min(original_w, int(box[2]))
            ymax = min(original_h, int(box[3]))
            
            # Draw rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label background and text
            label = f"{CLASSES[cls_idx]}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + w, ymin), color, -1)
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate and display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow('YOLO Live Inference', frame)
    
    # Press 'q' to exit the webcam loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()