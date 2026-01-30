# Stage 1: Builder
FROM python:3.12-slim AS builder

# Upgrade system packages, pip and install build dependencies
# hadolint ignore=DL3013
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build==1.3.0

# Set the working directory
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Build the wheel
RUN python -m build --wheel --outdir /wheels


# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Upgrade system packages
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Set the working directory
WORKDIR /home/appuser/app

# Upgrade pip, setuptools and wheel
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the wheel from the builder stage
COPY --from=builder /wheels /wheels

# Install the application wheel
RUN pip install --no-cache-dir /wheels/*.whl

# Set entrypoint to the wrapper script (which launches both NVFlare Client and API)
ENTRYPOINT ["python", "-m", "coreason_enclave.main"]
