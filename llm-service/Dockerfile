FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS llama_cpp_builder

ARG LLAMA_CPP_REPO=https://github.com/hanxiao/llama.cpp.git
ARG LLAMA_CPP_REF=master

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN CUDA_STUB_DIR="$(dirname "$(find /usr/local/cuda -path '*/stubs/libcuda.so' -print -quit)")" \
    && test -n "${CUDA_STUB_DIR}" \
    && ln -sf "${CUDA_STUB_DIR}/libcuda.so" "${CUDA_STUB_DIR}/libcuda.so.1" \
    && ln -sf "${CUDA_STUB_DIR}/libcuda.so" /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ln -sf "${CUDA_STUB_DIR}/libcuda.so" /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && ldconfig

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/targets/x86_64-linux/lib/stubs

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} ${LLAMA_CPP_REPO} /tmp/llama.cpp

RUN cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON \
    && cmake --build /tmp/llama.cpp/build --target llama-embedding llama-tokenize -j "$(nproc)"

RUN mkdir -p /tmp/llama-libs \
    && find /tmp/llama.cpp/build -name "*.so*" -exec cp -av {} /tmp/llama-libs/ \;

RUN mkdir -p /tmp/cuda-libs \
    && cp -av /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so* /tmp/cuda-libs/ \
    && cp -av /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so* /tmp/cuda-libs/ \
    && cp -av /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so* /tmp/cuda-libs/

FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

COPY --from=llama_cpp_builder /tmp/llama.cpp/build/bin/llama-embedding /usr/local/bin/llama-embedding
COPY --from=llama_cpp_builder /tmp/llama.cpp/build/bin/llama-tokenize /usr/local/bin/llama-tokenize
COPY --from=llama_cpp_builder /tmp/llama-libs/ /usr/local/lib/
COPY --from=llama_cpp_builder /tmp/cuda-libs/ /usr/local/cuda/lib64/

RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
