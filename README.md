# SPORTVISION
SPORTVISION  — A real-time football analytics project that uses AI and Computer Vision to track players and the ball during a match. It provides live insights such as player speed, cumulative distance covered, heatmaps, and an interactive dashboard for analyzing match performance using YOLO, FastAPI, and GPU acceleration.

It is an AI-based football analytics project developed to turn football match videos into meaningful insights and interactive visual analytics. The system processes football footage in real time, tracks players and the ball, and generates performance metrics that help analyze match activity more effectively.

The idea behind this project was to combine Computer Vision and Deep Learning with a web-based experience to create a system that not only detects objects but also provides useful match intelligence.

---

## Features

* Real-time player tracking
* Football detection and tracking
* Speed estimation
* Cumulative distance calculation
* Heatmap generation
* Dynamic analytics dashboard
* Live processed video preview
* Upload and analyze football videos
* GPU accelerated processing
* Real-time backend streaming

---

## Tech Stack

### Artificial Intelligence / Computer Vision

* YOLO
* OpenCV
* NumPy
* Deep Learning

### Backend

* FastAPI
* Uvicorn
* WebSockets

### Frontend

* HTML
* CSS
* JavaScript

### Hardware Acceleration

* NVIDIA CUDA
* PyTorch GPU support

---

## Project Workflow

Football Match Video / Live Stream

↓

Player and Ball Detection

↓

Object Tracking

↓

Speed and Distance Estimation

↓

Real-Time Processing

↓

Live Preview Generation

↓

Analytics Dashboard and Heatmaps

---

## System Capabilities

SPORTVISION can:

* Detect players and football in match footage
* Track player movement across frames
* Estimate player speed
* Calculate cumulative distance covered
* Generate heatmaps
* Display match insights through an interactive dashboard
* Stream processed video output in real time

---

## Installation

Clone the repository:

```bash
git clone <your-repository-link>
cd SPORTVISION
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python run_sportvision.py
```

Open in browser:

```text
http://127.0.0.1:8000
```

---

## Future Improvements

* Multi-camera support
* Team formation analysis
* Passing network visualization
* Tactical analysis
* Player comparison analytics
* Cloud deployment support

---

## Author

Kushagra Shukla
B.Tech CSE | Machine Learning and Computer Vision Enthusiast

If you found this project interesting, feel free to give it a star.

