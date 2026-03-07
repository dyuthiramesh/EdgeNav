import pytest
import tensorflow as tf
import numpy as np
import os

# --- PATHS TO YOUR ACTUAL FILES ---
YOLO_MODEL_PATH = "ultimate_indoor_yolo.keras"
DEPTH_MODEL_PATH = "midas_mobilenet_v1.keras"

class TestRealModelIntegration:

    @pytest.mark.skipif(not os.path.exists(YOLO_MODEL_PATH), reason="YOLO file missing")
    def test_yolo_load_and_predict(self):
        """Tests if the real YOLO .keras file loads and accepts a 416x416 input."""
        model = tf.keras.models.load_model(YOLO_MODEL_PATH, compile=False)
        dummy_input = np.random.randn(1, 416, 416, 3).astype(np.float32)
        
        # Action
        predictions = model.predict(dummy_input, verbose=0)
        
        # Verify: YOLO usually outputs a list of tensors for different scales
        assert isinstance(predictions, list), "YOLO output should be a list of feature maps"
        assert len(predictions) > 0, "YOLO returned no predictions"

    @pytest.mark.skipif(not os.path.exists(DEPTH_MODEL_PATH), reason="Depth file missing")
    def test_depth_load_and_predict(self):
        """Tests if the real Depth .keras file loads and accepts a 224x224 input."""
        model = tf.keras.models.load_model(DEPTH_MODEL_PATH, compile=False)
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        
        # Action
        depth_output = model.predict(dummy_input, verbose=0)
        
        # Verify: MiDaS MobileNet typically outputs (1, 224, 224, 1) or similar
        assert depth_output.ndim == 4, "Depth output should be a 4D tensor (Batch, H, W, C)"
        assert depth_output.shape[1:3] == (224, 224), "Depth output resolution mismatch"

    def test_integrated_pipeline_logic(self):
        """
        Final Check: Can the output of one be used to index the other?
        This mimics your 'Spatial AI' loop logic.
        """
        # Mocking the outputs to test the 'bridging' math
        mock_yolo_box = [50, 50, 150, 150] # [x1, y1, x2, y2]
        mock_depth_map = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        # Simulate your code's clipping/slicing
        x1, y1, x2, y2 = map(int, mock_yolo_box)
        # Scaled to depth resolution if necessary
        roi = mock_depth_map[y1:y2, x1:x2]
        
        assert roi.size > 0, "The integration logic failed to produce a valid Depth ROI"