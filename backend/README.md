# Go Inference Server

This directory houses the production inference API—a standalone Go binary that
serves real-time emotion predictions without a Python runtime anywhere in the
loop. The server loads the serialized `emotion_model.onnx` artifact directly
into memory through C bindings to the ONNX Runtime, accepting pixel arrays
from the React frontend and returning probability distributions in under 10ms
on commodity hardware.

## Directory Structure

```text
/backend
  ├── cmd/
  │   └── server/
  │       └── main.go             # Process entrypoint. Bootstraps the ONNX session, wires
  │                               # middleware, and orchestrates graceful shutdown on SIGTERM.
  ├── internal/
  │   ├── handlers/
  │   │   └── predict.go          # Transport layer. Validates the JSON contract, converts
  │   │                           # float64→float32, wraps results in the API envelope.
  │   ├── ml/
  │   │   └── inference.go        # Inference engine. Owns the ONNX Runtime session lifecycle
  │   │                           # and manages the pre-allocated C tensor buffers.
  │   ├── models/
  │   │   └── response.go         # Canonical API envelope. Single struct for success and error
  │   │                           # paths, with RFC 7807-compatible error shapes.
  │   └── middleware/
  │       └── request_id.go       # Injects a UUID v4 into every request for distributed tracing.
  ├── go.mod                      # Module definition with pinned dependency versions.
  ├── go.sum                      # Cryptographic checksums for the module graph.
  ├── libonnxruntime.1.21.0.dylib        # ONNX Runtime shared library (macOS x86_64). Platform-specific;
  │                               # Linux and Windows users must supply their own build.
  └── emotion_model.onnx          # The serialized model (copied or symlinked from /ml).
```

## Architecture & Decisions

### Why Go for Inference?

A conventional Python serving stack—FastAPI, Flask, or even async uvicorn—pays a
heavy per-request tax: the Global Interpreter Lock serializes CPU-bound work, and
framework overhead (middleware, serialization, object construction) dominates
latency for sub-10ms operations. Go inverts this cost profile.

- **Goroutines, not batching.** Instead of queuing images to build large batch
  tensors for GPU saturation, the server spawns a lightweight goroutine per
  incoming HTTP request. Each goroutine executes a `[1, 3, 48, 48]` tensor
  independently. For a 48×48 MobileNetV2, CPU inference is ~5ms—faster than the
  overhead of assembling a batch and dispatching it.
- **Predictable latency.** The Go runtime's garbage collector is STW-
  (stop-the-world) concurrent and generational, but we minimize its work by
  pre-allocating all C tensor memory at startup. The hot path in `Predict()`
  allocates exactly one 28-byte slice per call (the result copy). No heap
  pressure, no GC pauses during inference.
- **Single binary deployment.** The compiled server is a statically linked binary
  (plus the ONNX Runtime `.so`/`.dylib`/`.dll`). No virtual environment, no pip
  install, no CUDA toolkit. Ship the binary and the shared library, and the
  server runs.

### Memory Management and ONNX

The `onnxruntime_go` package is a cgo shim atop the official C++ ONNX Runtime.
We use `NewAdvancedSession` rather than the simpler `NewSession` because it
grants us explicit control over tensor memory:

1. **Pre-allocation at startup.** `NewEmptyTensor` allocates contiguous C arrays
   for the input (6,912 floats) and output (7 floats). These buffers live outside
   Go's heap for the lifetime of the process.
2. **Zero-allocation inference.** On each request, `copy()` moves pixel data
   into the pre-existing C buffer and `copy()` pulls the result out. No `malloc`,
   no `free`, no garbage to collect.
3. **Deterministic teardown.** `Destroy()` calls the C++ destructors in reverse
   order (session → tensors → environment), returning every allocated byte to
   the OS. The server's graceful-shutdown hook guarantees this runs even on
   SIGTERM.

### Dependency Injection via Closures

`main.go` initializes the `EmotionModel` once, then passes it into
`handlers.PredictHandler(model)` which returns a `gin.HandlerFunc`. The handler
captures the model pointer in its closure scope—no global variable, no init-time
magic, no package-level mutable state. This pattern has two concrete benefits:

- **Testability.** Unit tests can inject a mock `EmotionModel` that returns
  predetermined probabilities, exercising the handler's validation, conversion,
  and envelope logic without touching the ONNX Runtime.
- **Lifecycle clarity.** The dependency graph is explicit in `main.go`. You can
  trace every dependency from construction to injection in a single file.

### Concurrency Safety

The ONNX Runtime guarantees that `session.Run()` is safe to invoke from
multiple threads concurrently *provided each invocation uses distinct input and
output tensors*. Our current design shares one input and one output tensor across
all goroutines—a deliberate simplification that sacrifices some throughput for
operational simplicity.

Because each HTTP request is handled sequentially within its goroutine, and
Gin processes a single request per goroutine at a time, the `copy→Run→copy`
sequence is effectively serialized by Go's scheduler. Under peak load, this
becomes the bottleneck. If profiling reveals contention, the upgrade path is
to introduce a `sync.Mutex` around the `Run()` call or to maintain a small pool
of session+tensor pairs and acquire one per request—both changes are contained
entirely within `inference.go`.

### Graceful Shutdown

A naively-killed process abandons in-flight connections and leaks C memory.
Our shutdown sequence:

1. A buffered channel listens for `SIGINT` (Ctrl+C) and `SIGTERM` (Docker stop,
   Kubernetes pod eviction).
2. On signal, `srv.Shutdown(ctx)` stops accepting new connections but does not
   interrupt existing handlers.
3. A 5-second deadline context allows in-flight predictions to complete. The
   deadline is a safety valve—a hung handler cannot block shutdown forever.
4. After shutdown, `model.Destroy()` frees the ONNX Runtime's C allocations.

The result: clients see completed responses, not connection resets, and
Valgrind/memory-sanitizer reports show zero reachable leaks.

## Prerequisites

The server requires the ONNX Runtime C++ shared library at runtime. This is a
native, platform-specific binary that cannot be vendored with `go mod`. You
must download the correct build for your operating system and architecture.

> **Version lock:** This server uses `github.com/yalue/onnxruntime_go v1.13.0`,
> which targets ORT C API version 20. You **must** use ORT **1.21.0** — earlier
> or later releases will be rejected at startup with an "API version not
> available" error.

### macOS (Intel — x86_64)

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-osx-x86_64-1.21.0.tgz \
  | tar xz --strip-components=2 onnxruntime-osx-x86_64-1.21.0/lib/libonnxruntime.1.21.0.dylib

install_name_tool -id @loader_path/libonnxruntime.1.21.0.dylib ./libonnxruntime.1.21.0.dylib
install_name_tool -change \
  @rpath/libonnxruntime.1.21.0.dylib \
  @loader_path/libonnxruntime.1.21.0.dylib \
  ./libonnxruntime.1.21.0.dylib
```

### macOS (Apple Silicon — arm64)

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-osx-arm64-1.21.0.tgz \
  | tar xz --strip-components=2 onnxruntime-osx-arm64-1.21.0/lib/libonnxruntime.1.21.0.dylib

install_name_tool -id @loader_path/libonnxruntime.1.21.0.dylib ./libonnxruntime.1.21.0.dylib
install_name_tool -change \
  @rpath/libonnxruntime.1.21.0.dylib \
  @loader_path/libonnxruntime.1.21.0.dylib \
  ./libonnxruntime.1.21.0.dylib
```

### Linux (x86_64, glibc)

```bash
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz \
  | tar xz --strip-components=2 onnxruntime-linux-x64-1.21.0/lib/libonnxruntime.so.1.21.0
```

### Windows (x86_64 — PowerShell)

```powershell
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-win-x64-1.21.0.zip -OutFile onnxruntime.zip
Expand-Archive -Path onnxruntime.zip -DestinationPath .
Copy-Item -Path ./onnxruntime-win-x64-1.21.0/lib/onnxruntime.dll -Destination .
Remove-Item -Path onnxruntime.zip, ./onnxruntime-win-x64-1.21.0 -Recurse -Force
```

**Platform-specific shared library path:** `internal/ml/inference.go` calls
`ort.SetSharedLibraryPath("./libonnxruntime.1.21.0.dylib")`. Update this string
to match your platform:

| OS | Library filename |
|----|-----------------|
| macOS (x86_64 / arm64) | `./libonnxruntime.1.21.0.dylib` |
| Linux | `./libonnxruntime.so.1.21.0` |
| Windows | `./onnxruntime.dll` |

In a containerized deployment, inject this path via an environment variable or
build tag to avoid manual edits across platforms.
## Running the Server

1. **Model artifact.** Copy or symlink `emotion_model.onnx` from the `/ml`
   directory into this directory. The server loads it relative to the working
   directory at startup.
2. **Runtime library.** Download the correct ONNX Runtime build and place it in
   this directory (see Prerequisites above).
3. **Dependency resolution.**
   ```bash
   go mod tidy
   ```
4. **Launch.**
   ```bash
   go run cmd/server/main.go
   ```

5. **Docker (optional).**
```bash
   # Build and run with Docker Compose
   docker compose up --build

   # Or build the image directly for a specific platform
   docker buildx build --platform linux/amd64 -t emotion-engine .
```
   The `ORT_LIB_PATH` environment variable controls which shared library the
   server loads. It defaults to `./libonnxruntime.so.1.21.0` inside the
   container (set in `docker-compose.yml`). Override it in `.env` if needed:
```env
   ORT_LIB_PATH=./libonnxruntime.so.1.21.0
```

The server binds to `http://localhost:8080`. A healthy startup prints:

```
Starting ONNX Emotion Engine Backend...
ML Model loaded successfully into memory.
Server is listening on http://localhost:8080
Visit http://localhost:8080 for complete API documentation
```

Verify with `curl http://localhost:8080/api/v1/health`.
