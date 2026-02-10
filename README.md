# 🚦 AI-Powered Smart Traffic Management System

This project demonstrates an **AI-powered Smart Traffic Management System** designed to improve late-night road safety and reduce traffic collisions. It combines **YOLOv8 object detection** with a **real-time Pygame simulation** to detect vehicles from traffic footage and dynamically control traffic signals based on vehicle density.

Presented as part of **Late-Night Safety in Odisha using Machine Learning**.

---

## 1) Project Overview

The system uses Deep Learning and Computer Vision to:

- Detect vehicles from traffic camera footage
- Count vehicles in each direction
- Dynamically control traffic signals
- Improve late-night road safety and reduce collisions

### Core Subsystems

1. **AI Detection System** (`main.py`)
2. **Traffic Signal Simulation System** (`simulation.py`)

---

## 2) System Architecture

```text
Traffic Camera / Video
        ↓
OpenCV (Video Frames)
        ↓
YOLOv8 (Vehicle Detection)
        ↓
Vehicle Count per Direction
        ↓
Logic / Communication Layer
        ↓
Pygame Traffic Simulation
        ↓
Dynamic Traffic Signal Control
```

---

## 3) Features

- **AI Vehicle Detection:** Detects cars, trucks, buses, and motorcycles from video frames using YOLOv8.
- **Dynamic Signal Control:** Traffic lights adapt in real time based on directional vehicle density.
- **Simulation Dashboard:** Pygame visualization of traffic lights and vehicle counts by direction.
- **Safety-Oriented Design:** Prioritizes busiest lanes to reduce waiting time and potential collision risk, especially during low-traffic hours.

---

## 4) Tech Stack

- **Python**
- **YOLOv8 (Ultralytics)**
- **OpenCV** (video processing)
- **Pygame** (traffic simulation)
- **Regex + Subprocess** (AI ↔ simulation communication)

---

## 5) Project Structure

```text
Smart_Traffic_Management/
│
├── main.py
├── simulation.py
├── night_traffic.mp4
├── yolov8n.pt
├── requirements.txt
├── Smart_Traffic_Final.pptx
└── README.md
```

---

## 6) File-Wise Explanation

### `main.py` — AI Detection Module

**Purpose:** Deep Learning + Computer Vision core.

**What it does:**

- Reads `night_traffic.mp4`
- Extracts frames using OpenCV
- Uses YOLOv8 to detect vehicle classes:
  - car
  - truck
  - bus
  - motorcycle
- Counts vehicles by direction:
  - NORTH
  - SOUTH
  - EAST
  - WEST
- Outputs counts continuously

### `simulation.py` — Traffic Control & Visualization

**Purpose:** Controls and visualizes traffic signal logic.

**What it does:**

- Reads vehicle counts from `main.py`
- Selects the direction for green light
- Adjusts green duration based on traffic density
- Displays:
  - Signal state
  - Vehicle count per direction

### `night_traffic.mp4`

Sample late-night traffic footage used as input data for detection.

### `yolov8n.pt`

Pre-trained YOLOv8 model weights for fast and accurate object detection.

### `requirements.txt`

Dependency list for easy environment setup.

### `Smart_Traffic_Final.pptx`

Presentation deck describing objectives, architecture, and outcomes.

---

## 7) Working Flow

1. Traffic camera/video provides footage
2. OpenCV converts video stream into frames
3. YOLOv8 detects and counts vehicles
4. Counts are sent to the simulation module
5. Pygame updates signal timing dynamically
6. Busiest direction receives green signal priority

---

## 8) Why This Is a Deep Learning + CV Project

- Uses a CNN-based YOLOv8 detector
- Processes real video frames
- Performs real-time object detection
- Automates traffic-signal decision-making
- Targets practical road-safety impact

---

## 9) Setup and Run

### Install dependencies

```bash
pip install ultralytics opencv-python pygame
```

### Run simulation

```bash
python simulation.py
```

> Press `q` in the OpenCV window to stop detection.

---

## 10) One-Line Summary

This project integrates **Deep Learning (YOLOv8)** and **Computer Vision (OpenCV)** to build an AI-driven smart traffic management system that dynamically controls traffic signals using real-time vehicle detection and simulation.
