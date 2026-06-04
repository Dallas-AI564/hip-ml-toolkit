# Dockerfile for hip-ml-toolkit
# Requires: AMD GPU with ROCm support

FROM rocm/dev-ubuntu-22.04:6.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip3 install --no-cache-dir -e ".[dev,serve]"

ENTRYPOINT ["python3", "-m", "hipml"]
CMD ["info"]
