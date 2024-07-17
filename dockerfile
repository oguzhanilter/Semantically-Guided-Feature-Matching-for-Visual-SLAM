# Use the official Ubuntu 22.04 image as the base image
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive


# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config 

# Install additional dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran 

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone OpenCV and OpenCV contrib repositories
RUN git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git

# Create build directory and configure the build with CMake
RUN mkdir -p opencv/build && cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D BUILD_EXAMPLES=ON ..

# Build and install OpenCV
RUN cd opencv/build && make -j$(nproc) && make install

# Download and install LibTorch (C++ API for PyTorch)
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cpu && \
    mv libtorch /usr/local/libtorch

# Set environment variables for LibTorch and OpenCV
ENV Torch_DIR=/usr/local/libtorch
ENV CMAKE_PREFIX_PATH=${Torch_DIR}
ENV OpenCV_DIR=/usr/local/share/opencv4

# Clean up unnecessary files
RUN rm -rf /var/lib/apt/lists/*

# Entry point
CMD ["/bin/bash"]