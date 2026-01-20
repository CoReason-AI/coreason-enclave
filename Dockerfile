# Stage 1: Builder
FROM python:3.12-slim AS builder

# Upgrade system packages
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install build dependencies
RUN pip install --no-cache-dir build==1.3.0

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
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the wheel from the builder stage
COPY --from=builder /wheels /wheels

# Install the application wheel
RUN pip install --no-cache-dir /wheels/*.whl
