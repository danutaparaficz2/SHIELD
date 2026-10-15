FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. System packages & OpenFHE prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libomp-dev \
    python3 \
    python3-pip \
    python3-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Build and install OpenFHE core (v1.5.1 matches openfhe-python requirements)
RUN git clone https://github.com/openfheorg/openfhe-development.git --branch v1.5.1 --depth 1 && \
    mkdir -p openfhe-development/build && \
    cd openfhe-development/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_STATIC=ON \
        -DWITH_OPENMP=ON \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_UNITTESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# 3. Clone and install OpenFHE Python bindings (openfhe-python)
RUN git clone https://github.com/openfheorg/openfhe-python.git && \
    cd openfhe-python && \
    mkdir build && cd build && \
    cmake .. -DPYBIND11_FINDPYTHON=ON -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# 4. Clone FIDESlib GPU backend
RUN git clone https://github.com/FreeMark/openfhe-gpu-backend.git /workspace/openfhe-gpu-backend || true

WORKDIR /workspace
CMD ["/bin/bash"]
