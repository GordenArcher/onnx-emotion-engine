#
# Stage 1 — Download the ORT shared library for the target platform
#
# We use a separate stage for the download so that the curl/tar tooling never
# lands in the final image. The only artifact that crosses the stage boundary
# is the single .so file the Go server needs at runtime.
#
# TARGETARCH is injected automatically by `docker buildx build --platform`.
# Supported values: amd64, arm64.
# If you are building for a single known platform you can hard-code the URL
# and remove the ARG/RUN branching below.
FROM debian:bookworm-slim AS ort-downloader

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
ARG ORT_VERSION=1.21.0

# Map Docker's TARGETARCH values (amd64/arm64) to ORT's release naming
# convention (x64/arm64) and download the matching tarball.
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) ORT_ARCH="x64"  ;; \
      arm64) ORT_ARCH="arm64" ;; \
      *)     echo "Unsupported arch: ${TARGETARCH}" && exit 1 ;; \
    esac; \
    curl -fsSL \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
      -o /tmp/ort.tgz; \
    tar -xzf /tmp/ort.tgz -C /tmp; \
    # The versioned .so is the real library; the unversioned symlink is not
    # needed at runtime because we call SetSharedLibraryPath with the full name.
    cp /tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION} \
       /tmp/libonnxruntime.so.${ORT_VERSION}

#
# Stage 2 — Build the Go binary
#
# We use the official Go image so we get a known, reproducible toolchain.
# The build stage is thrown away after compilation; only the binary crosses
# into the final stage.
FROM golang:1.25-bookworm AS builder

WORKDIR /app

# Copy dependency manifests first so Docker can cache the module download
# layer independently of source changes.
COPY backend/go.mod backend/go.sum ./
RUN go mod download


# Copy the rest of the source tree and compile a fully static binary.
# CGO_ENABLED=1 is required because the onnxruntime_go binding uses cgo to
# call into the ORT C API. We link against glibc (not musl) to match the
# Debian-based runtime image.
COPY backend/ .
RUN CGO_ENABLED=1 GOOS=linux go build -o /app/server ./cmd/server/main.go

#
# Stage 3 — Minimal runtime image
#
# debian:bookworm-slim gives us glibc (required by the ORT .so) without the
# full Debian package set. The final image contains only:
#   - glibc and its dependencies (from the base image)
#   - the compiled Go binary
#   - the ORT shared library
#   - the ONNX model file
FROM debian:bookworm-slim AS runtime

# ca-certificates is needed if the server ever makes outbound HTTPS calls.
# libgomp1 is required by ORT's internal parallelism primitives.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG ORT_VERSION=1.21.0

# Pull in the ORT library from the downloader stage
COPY --from=ort-downloader /tmp/libonnxruntime.so.${ORT_VERSION} ./libonnxruntime.so.${ORT_VERSION}


# Pull in the compiled server binary from the builder stage
COPY --from=builder /app/server ./server

# The ONNX model file must be present at the path NewEmotionModel() expects.
# If you store models externally (S3, GCS) you would remove this line and
# mount the file at runtime instead.
COPY backend/emotion_model.onnx ./emotion_model.onnx

# Run as a non-root user. This is a hard requirement in most production
# environments and a security best practice everywhere else.
RUN useradd -r -u 1001 -g root appuser
USER appuser

EXPOSE 8080

# Pass the ORT library path via environment so the Go code can read it with
# os.Getenv("ORT_LIB_PATH") instead of having the filename hard-coded.
# If you prefer the hard-coded path, this ENV line can be removed.
ENV ORT_LIB_PATH=./libonnxruntime.so.${ORT_VERSION}

CMD ["./server"]
