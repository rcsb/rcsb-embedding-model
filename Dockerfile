# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy only necessary files for installation
COPY pyproject.toml README.md LICENSE.md ./
COPY src/ ./src/

# Install build dependencies, build, strip, and cleanup in one layer
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ binutils && \
    pip install --no-cache-dir -e . && \
    # Strip binaries immediately to reduce size before copying
    find /usr/local/lib/python3.12 -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type f -executable -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    # Remove unnecessary files in builder stage
    find /usr/local/lib/python3.12 -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12 -name "*.pyo" -delete && \
    # Remove build dependencies immediately
    apt-get purge -y gcc g++ binutils && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy the cleaned packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Final cleanup for runtime image
RUN find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name 'en*' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/share/i18n/locales -mindepth 1 -maxdepth 1 ! -name 'en_*' -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* 2>/dev/null || true

ENTRYPOINT ["inference"]
