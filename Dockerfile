FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    gfortran \
    libgtest-dev \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN cd /usr/src/gtest && cmake . && make && cp lib/*.a /usr/lib/
RUN pip3 install pyvista matplotlib pandas
WORKDIR /workspace
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build
CMD ["bash", "-c", "cd build && ctest --output-on-failure"]
