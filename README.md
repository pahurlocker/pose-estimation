# Human Pose Estimation

## Overview

This application performs multi-person 2D human pose estimation and person detection on video streams using a human pose estimation model and a person detection model that are deployed with the OpenVINO toolkit. The pose estimation model identifies up to 18 points: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles. The results of the pose estimation and person detection predictions are rendered in the application in real-time on a camera feed or pre-recorded video. In addition to the pose estimation, the inference time, number of people in the video, and the order of inference for each person are overlaid on the video.

People entering and exiting the video feed are saved in a database as a series of events that can be accessed via a JSON API. The application can be run on a CPU, GPU, or the Intel Neural Compute Stick 2 via the command line or a Flask-based web application. The web-based interface allows end users to toggle between a real-time camera feed and a pre-recorded demo video.

### Human Pose Estimation Model

The model is a multi-person 2D pose estimation network that uses a tuned MobileNet V1 as the feature extractor. The model is based on the OpenPose approach and was originally built using the Caffe framework. OpenPose is the first real-time model to jointly detect multiple body parts. The model has two outputs, PAF and a keypoint heatmap. The coordinates in the heatmap are scaled and used for the pose overlay frame by frame.

### Person Detection Model

The model is developed for pedestrian detection in retail scenarios. It is based on a MobileNetV2-like backbone that includes depth-wise convolutions (single convolution filter per input channel) to reduce computation for the 3x3 convolution block. It was also originally built using the Caffe framework. The model outputs the confidence for the prediction and bounding box coordinates. The application counts any result above a .4 confidence value as a person.

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

## Create SQLite Database

```bash
python create_db.py
```

## Run Demo in Window

```bash
Usage: demo.py [OPTIONS]

Options:
  --device-name TEXT  device to run the network on, CPU, GPU, or MYRIAD.
                      Default is CPU
  --precision TEXT    model precision FP16-INT8, FP16, or FP32. Default is
                      FP16-INT8
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

## References

https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/402-pose-estimation-webcam

https://docs.openvino.ai/latest/index.html

https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html
