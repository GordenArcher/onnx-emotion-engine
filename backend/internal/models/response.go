package models

import "time"

// EmotionData represents the successful payload returned when inference completes.
// I keep this separate from the envelope so the handler constructs just the data
// portion without caring about the outer API structure. If we ever add a field
// (say, a per-emotion confidence interval), the change stays in this struct and
// the envelope is untouched.
type EmotionData struct {
	DominantEmotion string             `json:"dominant_emotion"`
	Confidence      float64            `json:"confidence"`
	Probabilities   map[string]float64 `json:"probabilities"`
}

// BatchData wraps the results of a multi-face inference request. FaceCount is
// redundant with len(Results) but including it explicitly means the client
// doesn't have to count — and it makes log lines self-describing without
// having to deserialize the full results array.
type BatchData struct {
	FaceCount int   `json:"face_count"`
	Results   []any `json:"results"`
}

// Metadata holds operational telemetry for the request. I keep this separate
// from the data payload so monitoring tools can parse just the metadata without
// caring about the shape of the domain data. Adding a new metric (GPU
// utilization, queue wait time) is a one-field change here with zero impact
// on the rest of the codebase.
type Metadata struct {
	InferenceTimeMs float64 `json:"inference_time_ms"`
	ModelVersion    string  `json:"model_version"`
	Timestamp       string  `json:"timestamp"`
}

// APIResponse is the universal envelope for every HTTP response this server sends —
// success, error, documentation, health check, all of it. A single struct for
// every response shape means the frontend can write one parser and never
// special-case an endpoint. The `omitempty` tags ensure that error responses
// don't carry a null "data" field and success responses don't carry a null
// "errors" field, the wire format stays clean without any manual nil checks
// in the handlers.
type APIResponse struct {
	Status    string              `json:"status"`
	Message   string              `json:"message"`
	Data      any                 `json:"data,omitempty"`
	Errors    map[string][]string `json:"errors,omitempty"`
	Code      string              `json:"code"`
	RequestID string              `json:"request_id"`
	Metadata  Metadata            `json:"metadata"`
}

// API Documentation Models
// These structs power the self-documenting "/" endpoint. Using proper types
// instead of map[string]interface{} gives us compile-time guarantees that the
// documentation shape never drifts from what the handlers actually produce.
// If we ever generate an OpenAPI spec, these types are the source of truth.

// APIDocumentation is the top-level documentation payload. It describes the
// service, the model it wraps, every available route, and operational notes
// that don't fit neatly into a per-route description.
type APIDocumentation struct {
	Service     string     `json:"service"`
	Version     string     `json:"version"`
	Description string     `json:"description"`
	Model       ModelInfo  `json:"model"`
	Routes      []RouteDoc `json:"routes"`
	Notes       []string   `json:"notes"`
}

// ModelInfo captures the static contract between the Python training pipeline
// and the Go inference server. These values are not configuration — they are
// hard constraints baked into the exported ONNX graph. If the image size
// changes, the channel count changes, or the emotion class order changes,
// the model must be retrained and re-exported before any of these values
// can be updated. Treat this struct as immutable between model versions.
type ModelInfo struct {
	Architecture     string   `json:"architecture"`
	InputShape       string   `json:"input_shape"`
	InputLength      int      `json:"input_length"`
	PixelRange       string   `json:"pixel_range"`
	OutputShape      string   `json:"output_shape"`
	EmotionClasses   []string `json:"emotion_classes"`
	Framework        string   `json:"framework"`
	InferenceLatency string   `json:"inference_latency"`
}

// RouteDoc describes a single API endpoint in enough detail that a developer
// can integrate without reading any other documentation. The curl example field
// is not optional, a working copy-pasteable command is worth more than three
// paragraphs of prose.
type RouteDoc struct {
	Method          string            `json:"method"`
	Path            string            `json:"path"`
	Description     string            `json:"description"`
	Headers         map[string]string `json:"headers,omitempty"`
	Request         map[string]string `json:"request,omitempty"`
	Response        map[string]any    `json:"response,omitempty"`
	ResponseSuccess APIResponse       `json:"response_success,omitempty"`
	ResponseError   APIResponse       `json:"response_error,omitempty"`
	Example         string            `json:"example"`
}

// NewSuccessResponse constructs a valid success envelope for single-face inference.
// Using a constructor rather than initializing the struct directly in the handler
// ensures required fields, timestamp, model version, status — are never
// accidentally left blank. The handler should never need to know what
// "mobilenet-v2-v1.0.0" is called.
func NewSuccessResponse(requestID string, data any, inferenceMs float64) APIResponse {
	return APIResponse{
		Status:    "success",
		Message:   "Emotion inferred successfully",
		Data:      data,
		Code:      "EMOTION_INFERRED",
		RequestID: requestID,
		Metadata: Metadata{
			InferenceTimeMs: inferenceMs,
			ModelVersion:    "mobilenet-v2-v1.0.0",
			Timestamp:       time.Now().UTC().Format(time.RFC3339),
		},
	}
}

// NewBatchSuccessResponse constructs a valid success envelope for multi-face
// batch inference. The faceCount parameter is accepted explicitly rather than
// derived from len(results) because at the call site the handler already has
// the count, passing it avoids a redundant len() call and makes the
// constructor's intent explicit: you are telling it how many faces were
// processed, not asking it to figure it out.
//
// inference_time_ms here covers the full batch, not a single face. If you need
// per-face timing, add it to BatchFaceResult, don't put it in the envelope
// metadata where it would be ambiguous.
func NewBatchSuccessResponse(requestID string, results []any, inferenceMs float64, faceCount int) APIResponse {
	return APIResponse{
		Status:    "success",
		Message:   "Batch emotion inference completed",
		Data:      BatchData{FaceCount: faceCount, Results: results},
		Code:      "BATCH_EMOTION_INFERRED",
		RequestID: requestID,
		Metadata: Metadata{
			InferenceTimeMs: inferenceMs,
			ModelVersion:    "mobilenet-v2-v1.0.0",
			Timestamp:       time.Now().UTC().Format(time.RFC3339),
		},
	}
}

// NewErrorResponse constructs a consistent error envelope. The errs map accepts
// field-level validation errors so the frontend can display them next to the
// specific input that caused the problem, {"faces[2]": ["Expected 6912 floats,
// received 500"]} is more useful than a generic "validation failed" message.
// For non-validation errors (inference failures, internal errors), pass a single
// key that describes the failure domain ("inference", "model") rather than a
// field name.
func NewErrorResponse(requestID string, code string, message string, errs map[string][]string) APIResponse {
	return APIResponse{
		Status:    "error",
		Message:   message,
		Errors:    errs,
		Code:      code,
		RequestID: requestID,
		Metadata: Metadata{
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		},
	}
}
