#!/bin/bash

CONTAINER_NAME="orb3"
IMAGE_NAME="orbslam3-dev"

# Path of ORB_SLAM3 source on host
HOST_SRC_DIR="$(pwd)/../../DIP_ORB_SLAM3_ECE253"

# Check source folder exists
if [ ! -d "$HOST_SRC_DIR" ]; then
    echo "❌ ERROR: ORB_SLAM3 folder not found at:"
    echo "   $HOST_SRC_DIR"
    echo "Please clone ORB_SLAM3 into this directory."
    exit 1
fi

# Allow X11 connections
xhost +local:docker > /dev/null 2>&1

echo "🚀 Launching fresh ORB-SLAM3 development container with USB + X11..."

docker run -it --rm \
    --name $CONTAINER_NAME \
    --privileged \
    --device=/dev/bus/usb \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v "$HOST_SRC_DIR":/workspace/ORB_SLAM3 \
    -w /workspace/ORB_SLAM3 \
    $IMAGE_NAME \
    bash

