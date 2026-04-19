import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ==========================================
# 1. ARCHITECTURE (Must match your training)
# ==========================================
class SharpFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )
        self.conv_skip = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.refine(x + self.conv_skip(skip))

class MobileMiDaS_ReDWeb(nn.Module):
    def __init__(self):
        super().__init__()
        mnet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.enc1, self.enc2, self.enc3, self.enc4 = mnet[0:2], mnet[2:4], mnet[4:9], mnet[9:]
        self.bottleneck = nn.Conv2d(576, 128, kernel_size=1)
        self.fusion3 = SharpFusionBlock(128, 48)
        self.fusion2 = SharpFusionBlock(48, 24)
        self.fusion1 = SharpFusionBlock(24, 16)
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        out = self.bottleneck(s4)
        out = self.fusion3(out, s3)
        out = self.fusion2(out, s2)
        out = self.fusion1(out, s1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(out)

# ==========================================
# 2. PRE/POST PROCESSING
# ==========================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def run_inference(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileMiDaS_ReDWeb().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Normalization Stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    cap = cv2.VideoCapture(0)
    prev_depth = None
    alpha = 0.7 # Temporal smoothing factor

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # Pre-process
        frame_enhanced = apply_clahe(frame)
        img_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_rgb, (256, 256))
        img_tensor = torch.from_numpy(img_input).permute(2,0,1).float().to(device) / 255.0
        img_tensor = (img_tensor.unsqueeze(0) - mean) / std

        # Inference
        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()

        # Post-process: Temporal Smoothing
        if prev_depth is None: prev_depth = pred
        pred = alpha * pred + (1 - alpha) * prev_depth
        prev_depth = pred

        # Post-process: Normalize & Colorize
        depth_vis = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(heatmap, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ReDWeb Real-Time Depth", np.hstack((frame, heatmap)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference("mobile_midas_redweb_ep20.pth")