package models

import "time"

// EmotionData represents the successful payload returned when inference completes.
// I am keeping this separate from the envelope so the handler can easily construct
// just the data portion without worrying about the outer API structure.
type EmotionData struct {
	DominantEmotion string             `json:"dominant_emotion"`
	Confidence      float64            `json:"confidence"`
	Probabilities   map[string]float64 `json:"probabilities"`
}

// Metadata holds the operational telemetry for the request.
// Separating this makes it easy to add new metrics later (like GPU utilization or queue wait time)
// without touching the core data models.
type Metadata struct {
	InferenceTimeMs float64 `json:"inference_time_ms"`
	ModelVersion    string  `json:"model_version"`
	Timestamp       string  `json:"timestamp"`
}

// APIResponse is the strict, universal envelope for every HTTP response this server sends.
// By using a single struct for both success and error states, we guarantee frontend
// consumers always know how to parse the response, reducing frontend parsing bugs.
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
// These structs power the built-in documentation served at the root "/" endpoint.
// By defining them as proper types (rather than ad-hoc map[string]interface{}),
// we get compile-time guarantees that the documentation shape never drifts from
// what the handler constructs. The frontend can also generate TypeScript types
// from these definitions if we ever publish an OpenAPI spec derived from them.

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
// and the Go inference server. If any of these values change—image size, number
// of channels, emotion class order—the model must be retrained and re-exported.
// This struct serves as living documentation of that contract.
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
// can integrate with it without reading any other documentation. It includes
// the HTTP method, path, a human-readable description, full request/response
// schemas (using our actual APIResponse type), and a copy-pasteable curl example.
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

// NewSuccessResponse is a constructor function to cleanly build a valid success payload.
// I use constructors rather than initializing structs directly in handlers to ensure
// required fields (like timestamps and default statuses) are never accidentally left blank.
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

// NewErrorResponse is a constructor function to cleanly build a consistent error payload.
// I'm accepting a map of errors so the frontend can display validation errors
// next to specific form fields (e.g., {"pixels": ["array too short"]}).
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
