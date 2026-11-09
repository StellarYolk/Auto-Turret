# Auto-Turret

An open-vocabulary pan/tilt laser turret. You describe a target in plain words (for example "yellow tshirt" or "water bottle"), and the system searches for it in a live camera feed, locks on, and points a low-power laser at it using servo motors.

This is a project of the **DSAI Club, IIT Bhilai**.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Features](#features)
4. [Hardware](#hardware)
5. [Software Stack](#software-stack)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Colour Post-Filter](#colour-post-filter)
9. [Serial Protocol and Servo Control](#serial-protocol-and-servo-control)
10. [Calibration](#calibration)
11. [Simulation with PyBullet](#simulation-with-pybullet)
12. [Performance Notes](#performance-notes)
13. [Troubleshooting](#troubleshooting)
14. [License and Usage Policy](#license-and-usage-policy)

---

## Overview

Most detectors (YOLO, RF-DETR and similar) only recognise a fixed list of classes they were trained on. This project instead uses an **open-vocabulary detector**, so the target can be any object that can be described in text, without retraining.

The pipeline runs on a laptop with an NVIDIA GPU. The laptop handles vision and control, and a small microcontroller drives the servos over USB serial.

```
Camera -> Grounding DINO (text prompt) -> colour post-filter -> CSRT tracker
       -> pixel to servo angle mapping -> PySerial -> microcontroller -> pan/tilt servos + laser
```

## How It Works

1. **Prompt parsing.** The user types a description. If it contains a colour word, the colour is separated from the object phrase. For "black tshirt" the object phrase is "tshirt" and the colour is "black".
2. **Detection.** Grounding DINO (Swin-T) detects the object phrase in the frame and returns bounding boxes with confidence scores.
3. **Colour verification.** Grounding DINO does not reliably bind attributes to objects (a query like "yellow tshirt" can fire on anything yellow or any tshirt). Each candidate box is therefore cropped and checked in HSV colour space against the requested colour.
4. **Tracking.** Surviving boxes initialise OpenCV CSRT trackers. The heavy detector runs only every N frames (default 10), and the lightweight tracker fills the frames in between.
5. **Aiming.** The tracked box centre is converted into pan and tilt angles, using either a gain-based mapping or a fitted calibration.
6. **Actuation.** Angles are sent as a small text command over serial to the microcontroller, which drives the servos and switches the laser.

## Features

- Open-vocabulary targeting from plain text prompts
- Tracks multiple matching objects at once
- Detect-every-N-frames plus CSRT tracking for smooth, low-latency boxes
- HSV colour post-filter using the XKCD colour name list (about 950 names)
- Special handling for grayscale colours (black, white, gray) and for the red hue wraparound
- Serial protocol for pan, tilt and laser control
- Manual jog tool and least-squares affine calibration from pixel coordinates to servo angles
- Full PyBullet simulation of the turret (URDF model, simulated camera, joint control, random target respawning, video recording)

## Hardware

| Component | Notes |
|---|---|
| Laptop with NVIDIA GPU | Developed on an RTX 5060 laptop GPU. Any modern CUDA-capable GPU should work |
| USB webcam | Any UVC camera. Global shutter is helpful but not required |
| Microcontroller | Arduino Uno or Nano, or an ESP32 |
| 2 hobby servos | Metal gear (for example MG996R) recommended for pan and tilt |
| Pan/tilt bracket | 3D printed or an off-the-shelf kit |
| Servo driver (optional) | PCA9685 for cleaner multi-servo PWM |
| Laser module | Low-power 5 V diode module, under 5 mW, switched through a transistor or MOSFET, never directly from a GPIO pin |
| Power supply | Separate 5 V, 2 A or higher supply for the servos. Sharing a rail with the microcontroller causes brownouts and random resets |

## Software Stack

- Python 3.12
- PyTorch with CUDA
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (Swin-T checkpoint)
- OpenCV (contrib build, required for CSRT)
- PySerial
- Matplotlib (used only for the XKCD colour table)
- PyBullet (simulation)

## Installation

### 1. Clone and create an environment

```bash
git clone https://github.com/StellarYolk/Auto-Turret.git
cd Auto-Turret
python3 -m venv venv
source venv/bin/activate
```

### 2. CUDA toolkit

Grounding DINO compiles custom CUDA operators, so the full CUDA toolkit (not just the driver) must be installed, and its version should be compatible with your PyTorch build.

```bash
nvidia-smi                                          # driver and max supported CUDA version
python3 -c "import torch; print(torch.version.cuda)" # CUDA version PyTorch was built with
nvcc --version                                      # toolkit version (must exist)
```

If `nvcc` is missing, install the toolkit from NVIDIA's apt repository for your Ubuntu release, then set:

```bash
export CUDA_HOME=/usr/local/cuda   # or the versioned path, for example /usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

A minor version difference within the same CUDA major series (for example toolkit 13.2 with PyTorch built for 13.0) is generally fine for compiling the custom ops.

### 3. Grounding DINO and weights

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..

mkdir -p weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 4. Other dependencies

```bash
pip install opencv-contrib-python pyserial matplotlib numpy pillow pybullet
pip install transformers==4.35.2
```

Version notes:

- **transformers must be pinned to 4.35.2.** Newer releases removed `BertModel.get_head_mask`, which Grounding DINO's `bertwarper.py` relies on. Without the pin, model loading fails with `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`.
- If you also use `rfdetr`, it requires a newer `transformers`. Keep it in a separate virtual environment.
- Use `opencv-contrib-python`, not plain `opencv-python`. The CSRT tracker (`cv2.legacy.TrackerCSRT_create`) lives in the contrib build.

### 5. Offline mode (optional)

After the first successful run, the BERT text encoder is cached locally. Setting `HF_HUB_OFFLINE=1` skips network checks and speeds up start-up:

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
```

Disable this for the very first run so the weights can download.

## Usage

### Run the detector and tracker

```bash
python3 main.py
```

You will be asked for a target description. Examples:

```
enter the description of the target: phone
enter the description of the target: black tshirt
enter the description of the target: purple tshirt
```

Press `q` in the video window to quit.

Key settings near the top of `main.py`:

| Setting | Meaning | Default |
|---|---|---|
| `BOX_THRESHOLD` | Minimum confidence for a region to be kept as a detection | 0.45 |
| `TEXT_THRESHOLD` | Minimum token similarity for words to be attached to a box label | 0.35 |
| detect interval | Run the full detector every N frames, track in between | 10 |
| camera index | Index passed to `cv2.VideoCapture` | 1 |

Finding the right camera index:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

Some laptops expose metadata-only `/dev/videoN` nodes, so the working capture device may not be index 0.

## Colour Post-Filter

Open-vocabulary detectors tend to treat a prompt as a bag of related tokens, so compound prompts such as "yellow tshirt" can match anything yellow or any tshirt. Changing the prompt wording or adjusting thresholds helped only marginally, so a deterministic post-filter is used instead.

1. The colour word is extracted from the prompt by matching against the XKCD colour table (longest match wins, so "sky blue" beats "blue").
2. The colour is converted to an OpenCV HSV range.
3. Grounding DINO runs with the object phrase only.
4. Each candidate box is cropped, and the fraction of pixels inside the HSV range is computed. Boxes below the match ratio are discarded.

Implementation details that matter in practice:

- **Hue-centred matching.** For chromatic colours the hue window (about plus or minus 12) does the discriminating, while saturation and value use broad fixed floors. Anchoring saturation and value tightly to the reference swatch rejects real fabric, which varies with folds, shadows and lighting.
- **Grayscale colours.** Black, white and gray have no meaningful hue, so they are matched on value (brightness) only, with a low saturation cap.
- **Red wraparound.** Red sits at both ends of the hue circle (near 0 and near 179), so red uses two ranges whose masks are combined.
- **Match ratio threshold.** A threshold of roughly 0.15 to 0.3 works better than a strict value, since boxes include background and shadows. Tune it against your own camera and lighting.

## Serial Protocol and Servo Control

The laptop sends one text line per update:

```
P;<pan_angle>;<tilt_angle>;<laser_on>\n
```


The microcontroller sketch parses the line, clamps angles to 0 to 180, writes the servos and switches the laser pin. On the Python side, a small `TurretController` class wraps PySerial, waits for the board to reset after the port opens, clamps values, and on exit re-centres the servos and switches the laser off.

Finding the serial port:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

## Calibration

Calibration maps a pixel location in the camera frame to the servo angles that place the laser dot at that location. The basic version assumes a single operating distance and a camera mounted close to the laser.

1. Place markers at a known working distance (a 3x3 grid works well).
2. Use the manual jog tool (W, A, S, D) to move the servos until the laser lands on each marker.
3. Record the clicked pixel position and the servo angles for 8 to 10 points spread across the frame.
4. Fit an affine model by least squares:

```
pan  = a*x + b*y + c
tilt = d*x + e*y + f
```

The fitted coefficients become the pixel to angle mapping used by the main loop.

Limitations of the basic model: because the camera and laser are physically offset, parallax makes the mapping distance dependent. It is accurate near the calibration distance and degrades at other ranges. Smoothing (an exponential moving average on the commanded angles) is recommended to avoid jitter from small tracker fluctuations.

## Simulation with PyBullet

The turret can be developed and demonstrated without hardware.

```bash
python3 sim_turret.py
```

Files:

- `turret.urdf`: pan and tilt joints plus a camera link
- `sim_turret.py`: loads the model, renders a camera view from the turret, detects the target, drives the simulated joints

What the simulation shows:

- A yellow cube spawns at a random position. The turret detects it (green box), steers until the camera centre (red dot, which is where the turret is actually pointing) lies on the target centre, then the cube respawns elsewhere.
- Joint control uses PyBullet `POSITION_CONTROL` with explicit position and velocity gains.

Tuning for smooth motion (if the turret bobs around the target):

- Lower the proportional gain
- Increase velocity damping (`velocityGain`) and lower `positionGain`
- Apply exponential smoothing to the commanded angle
- Add a small pixel deadband near the centre

If the turret moves away from the target instead of towards it, flip the sign of the corresponding error term. The correct sign depends on joint axes and the camera convention, and is easiest to find by testing it.

Recording the simulation: render frames with `p.getCameraImage` using the matrices from `p.getDebugVisualizerCamera()` and write them with `cv2.VideoWriter`. Match the writer's fps to the real loop rate, and re-encode with ffmpeg (`-vcodec libx264 -pix_fmt yuv420p`) for broad playback compatibility.

Note on sim-to-real: the simulation validates the control logic and software architecture. Calibration values, gains and smoothing constants will still need re-tuning on real hardware. The open-vocabulary model also performs poorly on PyBullet's flat-shaded renders, so the simulation uses colour-blob detection rather than Grounding DINO.

## Performance Notes

- Grounding DINO is the heaviest part. Lowering the input resolution (for example `RandomResize([480], max_size=800)`) reduces latency substantially at some cost to small-object accuracy.
- A multi-second delay between the real scene and the displayed frame is usually **camera buffer backlog**, not the model. Set `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` and/or flush stale frames with `cap.grab()` before `cap.retrieve()`. Time the detector call separately to confirm where the delay actually is.
- If you need more speed, YOLO-World is a faster open-vocabulary alternative, with different accuracy characteristics.
- Start-up is slow because the BERT text encoder and the Swin backbone must load onto the GPU. This is a one-time cost per launch.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `AttributeError: ... get_head_mask` | Wrong `transformers` version. Install `transformers==4.35.2` |
| `Unable to locate package cuda-toolkit-...` | Repo lacks that exact version. Run `apt-cache madison cuda-toolkit` and install an available 13.x version |
| `pip install -e .` fails compiling CUDA ops | `CUDA_HOME` unset, or toolkit and PyTorch CUDA versions differ too much |
| `cv2.legacy` has no `TrackerCSRT_create` | Install `opencv-contrib-python` instead of `opencv-python` |
| `cv2.rectangle` argument errors | Box coordinates are tensors or floats. Convert to plain Python ints |
| Camera cannot be opened | Wrong `VideoCapture` index. Check `v4l2-ctl --list-devices` |
| Many deprecation warnings at start | Harmless. Filter `FutureWarning` and `UserWarning` at the top of `main.py` |
| No match for dark or gray colours | Grayscale needs the value-only branch, not hue matching |
| Servos twitch or the board resets | Servos sharing a power rail with the microcontroller. Use a separate supply |



## License and Usage Policy

This project was developed under the **DSAI Club, IIT Bhilai**.

If you wish to use, adapt or borrow any part of this code, you must first ask for and obtain permission from the DSAI Club, IIT Bhilai. Please contact the club before reusing any portion of this repository in your own work.

Third-party components (Grounding DINO, PyTorch, OpenCV, PyBullet and others) remain under their own respective licenses.