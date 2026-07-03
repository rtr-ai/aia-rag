FROM python:3.11-slim-bookworm AS llama_cpp_builder

ARG LLAMA_CPP_REPO=https://github.com/hanxiao/llama.cpp.git
ARG LLAMA_CPP_REF=master

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} ${LLAMA_CPP_REPO} /tmp/llama.cpp

RUN cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    && cmake --build /tmp/llama.cpp/build --target llama-embedding llama-tokenize -j "$(nproc)"

RUN mkdir -p /tmp/llama-libs \
    && find /tmp/llama.cpp/build -name "*.so*" -exec cp -av {} /tmp/llama-libs/ \;

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

RUN ldconfig

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
