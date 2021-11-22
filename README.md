# Human Pose Estimation

## Overview

This application performs human pose estimation on a video stream using pre-trained human pose estimation models with the OpenVINO library. It can be run on a CPU or the Intel Neural Compute Stick 2 via the command line or a Flask-based web application.

## Setup

### Python Dependencies

```bash
pip install -e .
```

### OpenVINO Installation

Instructions for installing OpenVINO on Linux, Windows, macOS, and Raspian OS

```link
https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html
```

## Create SQLite Databased

```bash
python create_db.py
```

## Run Demo in Window

```bash
Usage: demo.py [OPTIONS]

Options:
  --device-name TEXT  device to run the network on, CPU, GPU, or MYRIAD.
                      Default is CPU
  --video-url TEXT    use video instead of video camera, try
                      https://github.com/intel-iot-devkit/sample-
                      videos/blob/master/store-aisle-detection.mp4?raw=true.
                      Default uses video camera
  --help              Show this message and exit.
```

### Run with camera feed

```bash
python demo.py
```

### Run with sample video

```bash
python demo.py --video-url="https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4?raw=true"
```

## Run Flask Web Application

```bash
python main.py 
```

```link
http://127.0.0.1:5000/?device-name=CPU (or GPU, MYRIAD, or AUTO (default) for Intel Neural Compute Stick 2)
```

## Detection event API

```link
http://127.0.0.1:5000/detections?page=1&per-page=50
```

## Code citation

```link
https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/402-pose-estimation-webcam
```