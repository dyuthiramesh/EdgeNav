# EdgeNav: Assistive Navigation via Edge-Optimized Depth and Detection

EdgeNav is an end-to-end assistive navigation system designed to provide real-time spatial awareness for individuals with visual impairments. By combining monocular depth estimation and object detection on a Raspberry Pi 4, the system translates visual environments into actionable audio feedback.

---

## Overview

The project utilizes a shared-encoder architecture to perform two critical tasks simultaneously:
1. Depth Estimation: Predicting distances to obstacles using a MiDaS-style decoder.
2. Object Detection: Identifying specific household items and people using a YOLO-based head.

The system is optimized using the OpenVINO toolkit to achieve real-time inference speeds on resource-constrained edge hardware.

---

## Technical Architecture

### Model Design
* Encoder: MobileNetV3 (chosen for its high performance-to-latency ratio on mobile CPUs).
* Depth Decoder: A MiDaS-style lightweight decoder trained on the RedWeb dataset to ensure robust relative depth perception.
* Detection Head: YOLO architecture fine-tuned on a custom dataset combining HomeObjects and Person classes to prioritize indoor navigation safety.

### Optimization and Inference
* OpenVINO Integration: Models are converted to Intermediate Representation (IR) format and quantized for FP16/INT8 precision to maximize the Raspberry Pi 4 CPU capabilities.
* Piper TTS: A fast, local neural text-to-speech engine provides low-latency audio descriptions of the environment without requiring an internet connection.

---

## Results and Performance

### Inference Benchmarks (Raspberry Pi 4)
* Combined Inference Speed: Approximately 8-12 FPS using OpenVINO optimization (INT8 quantization).
* Latency: Average end-to-end latency (image capture to audio output) of ~150-200ms.
* Model Optimization: 75% reduction in model size and 4x speedup in inference compared to the original PyTorch implementation.

### Accuracy and Reliability
* Object Detection: Achieved a mean Average Precision (mAP) of 0.82 on the custom HomeObjects + Person dataset.
* Depth Estimation: Qualitative testing confirms reliable relative depth mapping for indoor obstacles within a 0.5m to 5m range.
* Audio Feedback: Piper TTS consistently delivers verbal cues in under 100ms post-inference, ensuring real-time relevance for the user.

---

## Tech Stack

* Frameworks: PyTorch (Training), OpenVINO (Inference)
* Hardware: Raspberry Pi 4 (4GB/8GB), OAK-D or USB Camera
* Datasets: RedWeb (Depth), HomeObjects + Custom Person Dataset (Detection)
* Audio: Piper TTS
* Languages: Python, C++

---

## Features

* Real-time Obstacle Avoidance: Continuous depth mapping to detect immediate path obstructions.
* Contextual Awareness: Recognition of critical indoor objects (chairs, tables, doors) and people.
* Edge Native: Entirely offline processing to ensure user privacy and reliability in various environments.
* Low Latency Audio: Piper TTS delivers descriptive feedback (e.g., 'Person at 2 meters, slightly left') with minimal delay.

---

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/EdgeNav.git
   cd EdgeNav

2. Set up OpenVINO on Raspberry Pi:
   Follow the official OpenVINO toolkit installation guide for Raspbian/Ubuntu.

3. Install Dependencies:
   pip install -r requirements.txt

4. Download Models:
   Place your .xml and .bin OpenVINO IR files in the /models directory.

5. Run the Application:
   python main.py --source 0 --device CPU

---

## Project Structure
```
EdgeNav/
├── models/             # OpenVINO IR files (.xml, .bin)
├── src/
│   ├── detection.py    # YOLO inference logic
│   ├── depth.py        # MiDaS decoder logic
│   ├── audio_engine.py # Piper TTS integration
│   └── utils.py        # Image processing & OpenVINO helpers
├── data/               # Custom dataset configuration files
├── main.py             # Main execution script
└── requirements.txt
```
---

## Acknowledgments

Developed as a Major Capstone Project at RV University. Special thanks to the faculty and teammates for the support in model training and hardware integration.
