import cv2
import numpy as np
import openvino as ov
import time
import csv
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEPTH_MODEL_XML = "best_openvino_depth_model/depth_model.xml"
OBJ_MODEL_XML   = "best_openvino_object_model/best.xml"
INPUT_SIZE_DEPTH = 256
INPUT_SIZE_OBJ = 416
REPORT_FILE = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class VisionProfiler:
    def __init__(self):
        self.data = {
            "preproc_ms": [],
            "depth_infer_ms": [],
            "obj_infer_ms": [],
            "total_latency_ms": [],
            "fps": []
        }

    def add_record(self, metrics):
        for key, value in metrics.items():
            if key in self.data: self.data[key].append(value)

    def save_to_csv(self):
        if not self.data["fps"]: return
        keys = self.data.keys()
        with open(REPORT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(zip(*[self.data[k] for k in keys]))
        print(f"\n[SUCCESS] Benchmark report saved to: {REPORT_FILE}")

def main():
    profiler = VisionProfiler()
    core = ov.Core()

    print("[INFO] Loading Models...")
    compiled_depth = core.compile_model(core.read_model(DEPTH_MODEL_XML), "AUTO")
    depth_req = compiled_depth.create_infer_request()
    
    compiled_obj = core.compile_model(core.read_model(OBJ_MODEL_XML), "AUTO")
    obj_req = compiled_obj.create_infer_request()

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting Benchmark (100 frames)...")

    frame_count = 0
    try:
        while frame_count < 100:
            ret, frame = cap.read()
            if not ret: break

            t_start = time.perf_counter()

            # --- PREPROCESSING ---
            t0 = time.perf_counter()
            # Depth Preproc
            depth_inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_inp = cv2.resize(depth_inp, (INPUT_SIZE_DEPTH, INPUT_SIZE_DEPTH)).astype(np.float32) / 255.0
            depth_inp = ((depth_inp - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]
            
            # Object Preproc (FIXED: Added transpose to NCHW)
            obj_inp = cv2.resize(frame, (INPUT_SIZE_OBJ, INPUT_SIZE_OBJ))
            obj_inp = obj_inp.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
            
            t_pre = (time.perf_counter() - t0) * 1000

            # --- INFERENCE ---
            t1 = time.perf_counter()
            depth_req.infer({0: depth_inp})
            t_depth = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            obj_req.infer({0: obj_inp})
            t_obj = (time.perf_counter() - t2) * 1000

            t_end = time.perf_counter()
            total_lat = (t_end - t_start) * 1000
            
            profiler.add_record({
                "preproc_ms": t_pre,
                "depth_infer_ms": t_depth,
                "obj_infer_ms": t_obj,
                "total_latency_ms": total_lat,
                "fps": 1.0 / (t_end - t_start)
            })

            cv2.imshow("Benchmarking", frame)
            if cv2.waitKey(1) == ord('q'): break
            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if len(profiler.data["total_latency_ms"]) > 0:
            profiler.save_to_csv()
            print("\n" + "="*40)
            print(f"Avg Total Latency: {np.mean(profiler.data['total_latency_ms']):.2f} ms")
            print(f"P95 Latency:       {np.percentile(profiler.data['total_latency_ms'], 95):.2f} ms")
            print(f"Avg FPS:           {np.mean(profiler.data['fps']):.2f}")
            print("="*40)
        else:
            print("[ERROR] No data collected. Check camera or model paths.")

if __name__ == "__main__":
    main()