package handlers

import (
	"fmt"
	"net/http"
	"time"

	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/ml"
	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/models"
	"github.com/gin-gonic/gin"
)

// PredictRequest mirrors the JSON contract our React client sends. The `json`
// struct tag dictates the exact field name that appears on the wire, but Go's
// encoding/json package will also accept a case-insensitive match, so "pixels",
// "Pixels", and "PIXELS" all deserialize into this struct. That leniency is
// convenient during development but can mask bugs if the frontend team
// accidentally renames the field—our explicit validation steps later catch the
// real semantic errors.
//
// Why float64? Go's json.Unmarshal parses every JSON number into a float64
// unless the target field is explicitly typed as json.Number. We accept the
// minor precision waste (64 bits on the wire, 32 bits in the model) because
// the conversion overhead is dwarfed by the ONNX inference itself.
type PredictRequest struct {
	Pixels []float64 `json:"pixels"`
}

// EmotionLabels is the rosetta stone that translates integer indices—the only
// thing the neural network understands—into human-readable emotion names.
// This mapping is not arbitrary; it is a hard contract with the Python training
// script. PyTorch's ImageFolder sorts class directories alphabetically using
// the host filesystem's locale, producing the order shown below. If this Go map
// and that Python sort order ever disagree, every prediction will be silently
// misattributed (e.g., "Happy" labeled as "Sad"), and the bug will look like a
// model accuracy problem when it is actually a data-contract violation. Treat
// this map as immutable unless you retrain the model.
var EmotionLabels = map[int]string{
	0: "Angry",
	1: "Disgust",
	2: "Fear",
	3: "Happy",
	4: "Neutral",
	5: "Sad",
	6: "Surprise",
}

// PredictHandler returns a Gin handler function that processes emotion-prediction
// requests. The function-signature itself is a closure factory: we accept the
// already-initialized EmotionModel as a parameter and return a gin.HandlerFunc
// that captures it. This is the canonical Go pattern for dependency injection
// without a DI framework—the handler owns a reference to its dependency, the
// dependency is never exposed as a global variable, and unit tests can inject a
// mock model that never touches the ONNX runtime.
func PredictHandler(model *ml.EmotionModel) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 1. Retrieve the request-scoped identifier
		// The RequestID middleware places a UUID into the Gin context before
		// this handler executes. Everything downstream—log lines, error
		// responses, and even our outgoing JSON payloads—carries this ID so
		// that a client reporting "my request failed" can be correlated to the
		// exact server-side log entry in seconds. If the ID is missing, we
		// fall back to a sentinel string rather than panicking; the handler
		// should degrade gracefully even when the middleware chain is misconfigured.
		requestID, exists := c.Get("request_id")
		if !exists {
			requestID = "unknown"
		}

		// 2. Deserialize and validate the payload
		// ShouldBindJSON performs two jobs in one call: it reads the request
		// body and unmarshals it into our PredictRequest struct, returning
		// the first error encountered. A failure here means the client sent
		// something that isn't valid JSON at all (e.g., a stray form-encoded
		// body, or missing the "pixels" key entirely).
		var req PredictRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			errResp := models.NewErrorResponse(
				requestID.(string),
				"INVALID_JSON",
				"Malformed JSON payload",
				map[string][]string{"body": {err.Error()}},
			)
			c.JSON(http.StatusBadRequest, errResp)
			return
		}

		// 3. Defend the tensor boundary
		// The ONNX model's input signature is [1, 3, 48, 48] = 6,912 float32
		// values. Feeding it an array of any other length will crash the
		// runtime or, worse, silently reinterpret memory. We enforce this
		// invariant at the API boundary with a hard length check. The error
		// message is deliberately verbose: it tells the caller both the
		// expected length and the length they actually supplied, turning a
		// 400 response into an actionable piece of documentation.
		if len(req.Pixels) != 6912 {
			errResp := models.NewErrorResponse(
				requestID.(string),
				"INVALID_PIXEL_DIMENSIONS",
				"Pixel array does not match required model input shape",
				map[string][]string{"pixels": {
					fmt.Sprintf("Expected array of length 6912 (48x48x3), received %d", len(req.Pixels)),
				}},
			)
			c.JSON(http.StatusBadRequest, errResp)
			return
		}

		// 4. Precision downgrade: float64 → float32
		// JSON gave us float64; the ONNX runtime expects float32. This loop
		// does an explicit, element-by-element conversion. Go does not allow
		// us to simply reinterpret the underlying slice as a different type
		// (unlike C++'s reinterpret_cast or Python's numpy.astype), so the
		// copy is unavoidable. With 6,912 elements, the cost is ~55 KB of
		// memory traffic—negligible compared to the ONNX graph execution that
		// follows.
		float32Pixels := make([]float32, len(req.Pixels))
		for i, val := range req.Pixels {
			float32Pixels[i] = float32(val)
		}

		// 5. Run inference against the ONNX runtime
		// The timer starts after all input validation and conversion, and
		// stops immediately after the model returns. This isolates the pure
		// computational cost of the neural network, which is the metric we
		// report to clients and monitor on our dashboards. Including JSON
		// serialization time in this measurement would conflate network-bound
		// work with CPU-bound work, making performance regressions harder to
		// diagnose.
		startTime := time.Now()

		probabilities, err := model.Predict(float32Pixels)
		if err != nil {
			// An inference error is rare but possible: the ONNX runtime can
			// fail if the model file was corrupted after startup, if memory
			// allocation fails under extreme load, or if an operator is
			// unsupported by the current runtime version. We surface these as
			// 500 errors because the client's request was well-formed and the
			// failure is entirely on our side.
			errResp := models.NewErrorResponse(
				requestID.(string),
				"ONNX_INFERENCE_ERROR",
				"Internal model execution failure",
				map[string][]string{"inference": {err.Error()}},
			)
			c.JSON(http.StatusInternalServerError, errResp)
			return
		}

		inferenceMs := float64(time.Since(startTime).Microseconds()) / 1000.0

		// 6. Argmax: find the winning emotion
		// The model outputs seven raw logits (unnormalized scores). We have
		// already applied softmax inside the ONNX graph (or in the Go runtime,
		// depending on the ml package implementation), so `probabilities` is
		// a true probability distribution that sums to 1.0. We perform a
		// linear scan for the maximum value—with only seven classes, the
		// overhead of a heap or sort is greater than a simple O(n) pass.
		maxIndex := 0
		maxProb := float64(0.0)
		probabilityMap := make(map[string]float64)

		for i, prob := range probabilities {
			p := float64(prob)
			probabilityMap[EmotionLabels[i]] = p
			if p > maxProb {
				maxProb = p
				maxIndex = i
			}
		}

		// 7. Assemble and return the response
		// The response shape is defined by models.EmotionData and carries the
		// dominant emotion, its confidence score, and the full probability
		// breakdown for all seven classes. Returning the full distribution
		// allows the frontend to render nuanced UI (e.g., a bar chart showing
		// that "Fear" and "Surprise" are neck-and-neck) without an extra
		// network round-trip.
		emotionData := models.EmotionData{
			DominantEmotion: EmotionLabels[maxIndex],
			Confidence:      maxProb,
			Probabilities:   probabilityMap,
		}

		successResp := models.NewSuccessResponse(
			requestID.(string),
			emotionData,
			inferenceMs,
		)

		c.JSON(http.StatusOK, successResp)
	}
}
