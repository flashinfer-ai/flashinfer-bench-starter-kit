#!/bin/bash

set -e
set -u

# Accept CUDA version as parameter (e.g., cu126, cu128, cu129)
CUDA_VERSION=${1:-cu128}

# Install torch with specific CUDA version first, followed by others in requirements.txt, and then others.
# This is to ensure that the torch version is compatible with the CUDA version.
pip3 install --force-reinstall torch --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip3 install -r /install/requirements.txt
pip3 install responses pytest scipy build cuda-python

# Install cudnn package based on CUDA version
if [[ "$CUDA_VERSION" == *"cu13"* ]]; then
  pip3 install --upgrade cuda-python==13.0
  pip3 install "nvidia-cudnn-cu13>=9.14.0.64"
else
  pip3 install --upgrade cuda-python==12.*
  pip3 install "nvidia-cudnn-cu12>=9.14.0.64"
fi

# Contest-specific packages
pip3 install tilelang cuda-tile cupti-python pandas cupy-cuda13x
