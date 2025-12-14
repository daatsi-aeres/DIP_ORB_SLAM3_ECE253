# Start from a basic Ubuntu 20.04 image
FROM ubuntu:20.04

# Set non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# 1. INSTALL SYSTEM & C++ BUILD DEPENDENCIES
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    # Librealsense Dependencies
    libssl-dev \
    libusb-1.0-0-dev \
    # OpenCV Dependencies
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    # Pangolin Dependencies
    libglew-dev \
    libboost-dev \
    libboost-thread-dev \
    libboost-filesystem-dev \
    libegl1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    # Other ORB-SLAM3 Dependencies
    libyaml-cpp-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. BUILD AND INSTALL OPENCV 4.5.4 FROM SOURCE
# --------------------------------------------------------------------------
WORKDIR /tmp
RUN git clone --depth 1 --branch 4.5.4 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.5.4 https://github.com/opencv/opencv_contrib.git
WORKDIR /tmp/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        ..
RUN make -j$(nproc)
RUN make install
RUN rm -rf /tmp/opencv /tmp/opencv_contrib

# 3. BUILD AND INSTALL PANGOLIN v0.6 FROM SOURCE
# --------------------------------------------------------------------------
WORKDIR /tmp
RUN git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git
WORKDIR /tmp/Pangolin/build
RUN cmake ..
RUN make -j$(nproc)
RUN make install
RUN rm -rf /tmp/Pangolin

# 4. BUILD AND INSTALL LIBREALSENSE SDK v2.50.0 FROM SOURCE
# --------------------------------------------------------------------------
WORKDIR /tmp
RUN git clone --depth 1 --branch v2.50.0 https://github.com/IntelRealSense/librealsense.git
WORKDIR /tmp/librealsense/build
RUN cmake ../ -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DFORCE_RSUSB_BACKEND=true
RUN make -j$(nproc)
RUN make install
RUN rm -rf /tmp/librealsense

# 5. INSTALL FINAL DEPENDENCY (EIGEN3) - THE SMART WAY
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y libeigen3-dev && rm -rf /var/lib/apt/lists/*

# 6. BUILD ORB_SLAM3
# --------------------------------------------------------------------------
# not here now
WORKDIR /workspace


# 7. INSTALL DEV & DEBUG UTILITIES
RUN apt-get update && apt-get install -y \
    gdb \
    vim \
    wget \
    curl \
    htop \
    net-tools \
    iputils-ping \
    unzip \
    less \
    && rm -rf /var/lib/apt/lists/*

# Set the final working directory and start a bash shell
WORKDIR /opt/ORB_SLAM3
CMD ["bash"]
