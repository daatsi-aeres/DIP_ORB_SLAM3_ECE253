# Adaptive-DIP-ORB-SLAM

**Enhancing ORB-SLAM3 Robustness Through Adaptive Digital Image Processing**
<p align="center">
  <img src="assets/Baseline_video_2.gif" width="800"/>
</p>


---

## Overview

This repository contains the source code and deployment environment for the academic project:

> **"Optimizing ORB-SLAM Robustness with an Adaptive Digital Image Processing Front-End"**
> *(ECE 253 – UC San Diego)*

The project integrates an **adaptive classical Digital Image Processing (DIP) pipeline** into the **front-end of ORB-SLAM3** to improve robustness under challenging real-world conditions, including:

* High Dynamic Range (HDR) illumination
* Sensor noise
* Motion blur

Unlike deep-learning-based approaches, this system relies on **lightweight classical DIP techniques**, making it suitable for real-time operation on embedded and robotic platforms.

---

## Key Contributions

* Adaptive image enhancement pipeline tightly coupled with ORB-SLAM3 front-end
* Improved tracking robustness under illumination and motion degradation
* Fully reproducible **Docker-based build and runtime environment**
* RealSense D435i stereo–inertial integration

---

## System Requirements

### Hardware

* **Camera:** Intel RealSense D435i
  *(Required for stereo + IMU-guided deblurring)*
* **Host OS:** Linux (tested on Ubuntu 20.04)

### Software

* **Docker Engine** (for environment isolation)
* **Git**
* **X11 forwarding** (for ORB-SLAM3 GUI inside Docker)

---

## Repository Structure

```text
Adaptive-DIP-ORB-SLAM/
├── ORB_SLAM3/
│   ├── setup_resources/
│   │   ├── Dockerfile          # Docker build environment
│   │   └── dev.sh              # Host-side container launcher
│   ├── run_realsense.sh        # Runtime script (inside container)
│   ├── realsense_d435i.yaml    # Camera & pipeline configuration
│   ├── Vocabulary/
│   ├── Examples/
│   └── ... (ORB-SLAM3 source)
```

---

## Docker-Based Setup (Recommended)

All dependencies are encapsulated in Docker to ensure **reproducibility** across systems.

### Step 1: Clone the Repository

```bash
git clone git@github.com:daatsi-aeres/DIP_ORB_SLAM3_ECE253.git
cd DIP_ORB_SLAM3_ECE253
```

---

### Step 2: Build the Docker Image

Navigate to the Docker setup directory:

```bash
cd DIP_ORB_SLAM3_ECE253/setup_resources
docker build -t orbslam3-dev .
```

**Note:** This step may take significant time. The Dockerfile builds the following from source:

* OpenCV 4.5.4
* Pangolin v0.6
* Librealsense SDK v2.50.0

---

### Step 3: Launch the Development Container

Verify that the image was built successfully:

```bash
docker images | grep orbslam3-dev
```

Then launch the container:

```bash
./dev.sh
```

You should now be inside the container at:

```text
/workspace/DIP_ORB_SLAM3_ECE253
```

#### What `dev.sh` Does

* Enables X11 forwarding for GUI visualization
* Launches Docker in privileged mode
* Mounts USB devices for RealSense access
* Mounts ORB-SLAM3 source into the container workspace

---

### Debugging `dev.sh`

| Issue                        | Fix                                                                    |
| ---------------------------- | ---------------------------------------------------------------------- |
| `ORB_SLAM3 folder not found` | Run `dev.sh` from `ORB_SLAM3/setup_resources`                          |
| `Cannot open display`        | Run `xhost +local:docker` on the host                                  |
| USB permission denied        | Ensure Docker has device access and your user is in the `docker` group |

---

## Build the Source Code

Inside the Docker container:

```bash
mkdir build
cd build
cmake .. -D CMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The build generates the ORB-SLAM3 executables under `Examples/`.

---

## ▶Run the Adaptive DIP-SLAM Pipeline

From inside the container:

```bash
cd ..
chmod +x run_realsense.sh
./run_realsense.sh
```

The ORB-SLAM3 viewer should launch, and the Adaptive DIP-enhanced pipeline will begin processing stereo–inertial data from the RealSense D435i.

---

### Debugging `run_realsense.sh`

| Issue                 | Fix                                                      |
| --------------------- | -------------------------------------------------------- |
| Executable not found  | Ensure build completed successfully (`Examples/Stereo/`) |
| Vocabulary missing    | Check `ORB_SLAM3/Vocabulary/ORBvoc.txt`                  |
| Settings file missing | Ensure `realsense_d435i.yaml` exists                     |
| Camera not detected   | Verify USB device mapping in `dev.sh`                    |

---

## Results

The adaptive DIP front-end improves ORB-SLAM3 tracking stability in:

* High-contrast HDR scenes
* Motion-blurred sequences
* Noisy sensor conditions

Quantitative and qualitative results are discussed in the accompanying course report.

---

## Acknowledgments

* ORB-SLAM3 authors
* Intel RealSense SDK
* UC San Diego – ECE 253

---

## License

This repository follows the licensing terms of **ORB-SLAM3**. Please consult the original project for commercial usage restrictions.

---

## Contact

For questions or collaboration inquiries, feel free to open an issue or contact the repository maintainer.
