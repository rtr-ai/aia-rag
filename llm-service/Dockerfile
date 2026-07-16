ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS llama_cpp_builder

ARG LLAMA_CPP_REPO=https://github.com/hanxiao/llama.cpp.git
ARG LLAMA_CPP_REF=96d58cc68ee4d59804b17ceeb3c46541e2d2a05e

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN DISCOVERED_CUDA_STUB_DIR="$(dirname "$(find /usr/local/cuda -path '*/stubs/libcuda.so' -print -quit)")" \
    && test -n "${DISCOVERED_CUDA_STUB_DIR}" \
    && ln -sf "${DISCOVERED_CUDA_STUB_DIR}/libcuda.so" "${DISCOVERED_CUDA_STUB_DIR}/libcuda.so.1" \
    && if [ -f /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so ]; then ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1; fi \
    && ln -sf "${DISCOVERED_CUDA_STUB_DIR}/libcuda.so" /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ln -sf "${DISCOVERED_CUDA_STUB_DIR}/libcuda.so" /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && ldconfig

ENV CUDA_STUB_DIR=/usr/local/cuda/targets/x86_64-linux/lib/stubs
ENV LIBRARY_PATH=${CUDA_STUB_DIR}:/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=${CUDA_STUB_DIR}:/usr/local/cuda/lib64/stubs

RUN git init /tmp/llama.cpp \
    && git -C /tmp/llama.cpp remote add origin ${LLAMA_CPP_REPO} \
    && git -C /tmp/llama.cpp fetch --depth 1 origin ${LLAMA_CPP_REF} \
    && git -C /tmp/llama.cpp checkout --detach FETCH_HEAD

RUN cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        -DGGML_CUDA=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_STUB_DIR} -Wl,-rpath-link,${CUDA_STUB_DIR} -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L${CUDA_STUB_DIR} -Wl,-rpath-link,${CUDA_STUB_DIR} -lcuda" \
    && cmake --build /tmp/llama.cpp/build --target llama-embedding llama-tokenize -j "$(nproc)"

RUN mkdir -p /tmp/llama-libs \
    && find /tmp/llama.cpp/build -name "*.so*" -exec cp -av {} /tmp/llama-libs/ \;

RUN mkdir -p /tmp/cuda-libs \
    && cp -av /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so* /tmp/cuda-libs/ \
    && cp -av /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so* /tmp/cuda-libs/ \
    && cp -av /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so* /tmp/cuda-libs/

FROM python:3.11-slim-bookworm

ARG PYTORCH_VERSION=2.11.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

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

RUN pip install --no-cache-dir "torch==${PYTORCH_VERSION}" \
        --index-url "${PYTORCH_INDEX_URL}"

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import torch; assert torch.__version__.startswith('${PYTORCH_VERSION}+cu128'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda"

EXPOSE 8000

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
