package ml

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// EmotionModel encapsulates a fully-initialized ONNX Runtime inference session
// together with its pre-allocated input and output tensors. Bundling these
// three objects into a single struct serves two purposes:
//
//  1. It hides the complexity of the ONNX Runtime C API behind a two-method
//     interface (Predict and Destroy), so the HTTP handler never touches raw
//     tensors or session handles.
//  2. It enables us to pre-allocate tensor memory once at startup and reuse it
//     for every subsequent prediction. Allocating and freeing C memory on every
//     HTTP request would introduce allocator contention under load and generate
//     unnecessary GC pressure on the Go side.
//
// The struct is not safe for concurrent use by multiple goroutines. Our HTTP
// server handles concurrency at the handler level—each request receives its own
// goroutine, but all goroutines share this single model instance. If we ever
// observe contention, the fix is a sync.Mutex around Predict, not per-request
// model instantiation.
type EmotionModel struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// NewEmotionModel bootstraps the entire ONNX inference pipeline. It is designed
// to be called exactly once during server startup. The initialization sequence
// is deliberately sequential and blocking—we want to fail fast and loudly if
// the model file is missing or the runtime library cannot be loaded, rather
// than discovering the problem on the first user request.
//
// The function performs four distinct setup steps:
//  1. Load the platform-specific ONNX Runtime shared library.
//  2. Allocate memory for the fixed-shape input and output tensors.
//  3. Parse the ONNX model file and create an execution session.
//  4. Wire the input/output tensors to the named graph endpoints.
func NewEmotionModel(modelPath string) (*EmotionModel, error) {
	// 1. Load the ONNX Runtime native library
	// The Go bindings for ONNX Runtime are a thin wrapper around a C shared
	// library. This line tells the wrapper where to find that library on disk.
	// The filename is platform-dependent and must match the OS where this
	// server is deployed:
	//
	//   macOS:           "./libonnxruntime.dylib"
	//   Linux (glibc):   "./libonnxruntime.so"
	//   Linux (musl):    "./libonnxruntime.so"  (separate build required)
	//   Windows:         "./onnxruntime.dll"
	//
	// If this path is wrong, the process will panic with a "failed to load
	// shared library" error. In a containerized deployment, this library
	// should be copied into the Docker image at build time or mounted as a
	// volume at a known, stable path.
	ort.SetSharedLibraryPath("./libonnxruntime.dylib")

	// 2. Define tensor shapes
	// The shape constants are not configurable at this layer; they are a hard
	// contract with the Python export script. Changing the image size, the
	// number of channels, or the number of emotion classes requires retraining
	// and re-exporting the model, not just tweaking these constants.
	//
	// Batch size is locked to 1. Web servers handle concurrent requests by
	// spawning goroutines, not by stacking multiple images into a single
	// tensor. A batch size of 1 keeps memory predictable (one input tensor =
	// 6,912 floats = ~27 KB) and avoids the complexity of dynamic batching,
	// which would require an external queue and a separate batching goroutine.
	inputShape := ort.NewShape(1, 3, 48, 48)
	outputShape := ort.NewShape(1, 7) // one probability per emotion class

	// 3. Allocate tensor memory
	// NewEmptyTensor allocates a contiguous block of C memory sized to hold
	// the entire tensor. This memory lives outside Go's garbage-collected heap
	// and must be explicitly freed via the Destroy() method. Pre-allocating
	// once and reusing the same buffers across thousands of predictions avoids
	// the overhead of repeated cgo calls to malloc/free, which are
	// significantly more expensive than Go allocations due to the cgo context
	// switch.
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate input tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate output tensor: %w", err)
	}

	// 4. Create the ONNX session
	// NewAdvancedSession parses the model file, validates the computation
	// graph, and prepares the runtime for execution. The string arguments
	// "input" and "output" correspond to the input_names and output_names we
	// specified in torch.onnx.export. A mismatch here (e.g., the Python script
	// exported with name "pixels" but we pass "input") produces a cryptic
	// runtime error, so these strings are effectively an implicit API contract
	// between the Python and Go codebases.
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},                   // must match Python's input_names
		[]string{"output"},                  // must match Python's output_names
		[]ort.ArbitraryTensor{inputTensor},  // bind the pre-allocated input
		[]ort.ArbitraryTensor{outputTensor}, // bind the pre-allocated output
		nil,                                 // no custom session options
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &EmotionModel{
		session: session,
		input:   inputTensor,
		output:  outputTensor,
	}, nil
}

// Predict executes a forward pass through the neural network and returns the
// seven-class probability distribution over emotions. It is the hot path of
// this entire service—every HTTP request to /api/v1/predict funnels through
// this method—so we keep it as lean as possible.
//
// The caller is responsible for ensuring that `pixels` is exactly 6,912
// elements long. We re-validate this invariant defensively inside the method
// because a slice-to-C-memory copy with mismatched lengths can cause a buffer
// overflow inside the ONNX runtime, which manifests as a segmentation fault
// that kills the entire Go process rather than returning a graceful error.
func (m *EmotionModel) Predict(pixels []float32) ([]float32, error) {
	// Reject invalid input before touching C memory. The HTTP handler should
	// have already performed this check, but defense in depth is cheap
	// insurance against future refactors that might bypass the handler's
	// validation logic.
	if len(pixels) != 6912 {
		return nil, errors.New("expected exactly 6912 pixels (48x48x3)")
	}

	//  Copy Go memory → C memory
	// The ONNX Runtime operates directly on the C-allocated tensor buffer.
	// Go's garbage collector cannot see this memory, so we must explicitly
	// copy our slice data into it. copy() is a compiler intrinsic on most
	// platforms and compiles to a fast memmove, keeping the overhead of this
	// step in the low-microsecond range.
	copy(m.input.GetData(), pixels)

	// Execute the ONNX computation graph
	// This single call crosses the cgo boundary into the ONNX Runtime's C++
	// engine, which walks the model graph node-by-node, executing each
	// operator (convolutions, batch normalizations, ReLUs, the final softmax)
	// and writing the result directly into our pre-allocated output tensor.
	// The runtime is single-threaded by default; enabling intra-op parallelism
	// would require setting session options (the nil argument in
	// NewAdvancedSession) and is an optimization left for when profiling
	// proves it necessary.
	if err := m.session.Run(); err != nil {
		return nil, fmt.Errorf("ONNX session run failed: %w", err)
	}

	// Copy C memory → Go memory
	// We return a Go-owned copy of the output rather than a slice pointing
	// directly into the C buffer. This is critical for thread safety: if a
	// second HTTP request calls Predict() while the first request is still
	// reading the result, the copy guarantees isolation. Returning a direct
	// reference to m.output.GetData() would create a data race and non-
	// deterministic behavior under concurrent load.
	outputData := m.output.GetData()
	result := make([]float32, 7)
	copy(result, outputData)

	return result, nil
}

// Destroy releases all C-allocated resources back to the operating system.
// This method must be called exactly once, during graceful server shutdown.
// Skipping it leaks memory that the Go garbage collector cannot reach—the
// ONNX Runtime allocates its tensor buffers and internal graph executor state
// with plain malloc, not with Go's allocator. In practice, the OS reclaims
// everything when the process exits, so the leak only matters if you are
// running multiple model load/unload cycles (e.g., hot-reloading models in a
// long-running process) or if you want clean Valgrind/memory-sanitizer output
// during development.
func (m *EmotionModel) Destroy() {
	m.session.Destroy()
	m.input.Destroy()
	m.output.Destroy()
	ort.DestroyEnvironment()
}
