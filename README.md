# Human Pose Estimation

## Overview

This application performs human pose estimation on a video stream using pre-trained human pose estimation models with the OpenVINO library. It can be run on a CPU or the Intel Neural Compute Stick 2 via the command line or a Flask-based web application.

## Setup

### Python Dependencies

```
python setup.py install
```

### OpenVINO Installation

Instructions for installing OpenVINO on Linux, Windows, macOS, and Raspian OS

https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html

## Run Demo in Window

```
Usage: demo.py [OPTIONS]

Options:
  --device-name TEXT  device to run the network on, CPU or MYRIAD. Default is
                      CPU
  --help              Show this message and exit.
```

## Run Flask Web Application

```
python main.py 

http://127.0.0.1:5000/?device-name=CPU (or MYRIAD for Intel Neural Compute Stick 2)
```