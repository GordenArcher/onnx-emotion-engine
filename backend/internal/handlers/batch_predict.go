package handlers

import (
	"fmt"
	"net/http"
	"time"

	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/ml"
	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/models"
	"github.com/gin-gonic/gin"
)

// BatchPredictRequest mirrors the JSON contract for multi-face inference.
// Instead of a single flat pixel array, the client sends a 2D array, one
// inner array per detected face. The outer slice length is variable (1 to N
// faces), but every inner slice must be exactly 6,912 floats. We validate
// this per-element rather than up-front so the error message can tell the
// client exactly which face index failed, not just "something is wrong."
//
// The same float64 reasoning from PredictRequest applies here: JSON numbers
// unmarshal as float64, and we convert to float32 before crossing the cgo
// boundary. The extra memory cost scales with face count, but even at 10
// simultaneous faces it's under 600 KB, well within budget.
type BatchPredictRequest struct {
	Faces [][]float64 `json:"faces"`
}

// BatchFaceResult holds the inference output for a single face within a batch.
// We carry the original index so the frontend can map each result back to the
// bounding box it came from. Without the index, the client has to assume the
// response array is in the same order as the request array — which it is, but
// making it explicit prevents a whole class of subtle UI bugs where face 2's
// emotion label ends up rendered over face 1's bounding box.
type BatchFaceResult struct {
	Index           int                `json:"index"`
	DominantEmotion string             `json:"dominant_emotion"`
	Confidence      float64            `json:"confidence"`
	Probabilities   map[string]float64 `json:"probabilities"`
}

// BatchPredictHandler handles POST /api/v2/predict/batch.
//
// The architectural argument for a batch endpoint over N parallel single-face
// calls is straightforward: N parallel calls create N round trips, N JSON
// parse cycles, and N competing goroutines that all want the same ONNX session.
// A single batch call collapses that to one round trip and one sequential pass
// through the session. The session itself is not thread-safe for concurrent
// Run() calls sharing the same tensors, so parallel single-face calls would
// require a mutex anyway, giving us the worst of both worlds: multiple round
// trips and serialized execution. The batch endpoint gives us one round trip
// and serialized execution by design.
//
// We deliberately keep the batch size unbounded at the API layer but rely on
// the client (MediaPipe) to cap it naturally, the browser's face detector
// won't return more faces than are actually visible. In practice this is 1-5.
// If we ever need server-side protection, a max-faces check is a two-line
// addition at the top of this handler.
func BatchPredictHandler(model *ml.EmotionModel) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 1. Pull the request-scoped trace ID
		// Same pattern as PredictHandler, every response carries this ID
		// so a client can quote it when reporting a bug and we can find the
		// exact log line in seconds.
		requestID, exists := c.Get("request_id")
		if !exists {
			requestID = "unknown"
		}

		// 2. Deserialize the payload
		// A missing or malformed "faces" key produces a clear 400 rather than
		// a panic downstream. The most common mistake here will be a client
		// accidentally sending {"pixels": [...]} (the v1 format) to the v2
		// endpoint, ShouldBindJSON will succeed but req.Faces will be nil,
		// which we catch in the next step.
		var req BatchPredictRequest
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

		// 3. Reject empty or nil face arrays
		// An empty batch is almost certainly a client bug — MediaPipe should
		// only call this endpoint when it has at least one detected face. We
		// surface it as a 400 rather than returning an empty success response,
		// because silently succeeding with zero results would be harder to
		// debug than a clear error.
		if len(req.Faces) == 0 {
			errResp := models.NewErrorResponse(
				requestID.(string),
				"EMPTY_BATCH",
				"Faces array must contain at least one pixel array",
				map[string][]string{"faces": {"received empty array"}},
			)
			c.JSON(http.StatusBadRequest, errResp)
			return
		}

		// 4. Validate every face's pixel array length before touching the model
		// We do a full pre-flight pass rather than validating lazily inside the
		// inference loop. The reason: if face index 4 of 5 has the wrong length,
		// we want to reject the entire request upfront rather than running
		// inference on faces 0-3 and then failing. Partial results with a mixed
		// success/error state would force the client to implement partial-failure
		// handling logic that it shouldn't need for a simple validation error.
		for i, face := range req.Faces {
			if len(face) != 6912 {
				errResp := models.NewErrorResponse(
					requestID.(string),
					"INVALID_PIXEL_DIMENSIONS",
					"One or more faces have invalid pixel array length",
					map[string][]string{
						fmt.Sprintf("faces[%d]", i): {
							fmt.Sprintf("Expected 6912 floats (48x48x3), received %d", len(face)),
						},
					},
				)
				c.JSON(http.StatusBadRequest, errResp)
				return
			}
		}

		// 5. Run inference sequentially across all faces
		// Sequential is the right call here, not parallel. The ONNX session
		// owns a single pair of pre-allocated C tensor buffers. Concurrent
		// Run() calls on those buffers would be a data race, two goroutines
		// writing into the same C memory simultaneously. The fix is not a
		// mutex (that just serializes with extra overhead) but to embrace
		// sequential execution: copy face N in, run, copy result out, move
		// to face N+1. For 1-5 faces at ~4ms each, the total batch latency
		// is 4-20ms, still well under any meaningful user-perception threshold.
		startTime := time.Now()
		results := make([]BatchFaceResult, 0, len(req.Faces))

		for i, face := range req.Faces {
			// Convert float64 → float32 for the ONNX runtime.
			// Same reasoning as in PredictHandler, JSON gives us float64,
			// the model wants float32, explicit conversion is the only option.
			float32Pixels := make([]float32, len(face))
			for j, val := range face {
				float32Pixels[j] = float32(val)
			}

			probabilities, err := model.Predict(float32Pixels)
			if err != nil {
				// An inference failure on one face fails the entire batch.
				// We could return partial results (faces 0 to i-1) but that
				// pushes complex partial-failure handling onto the client.
				// A clean 500 is easier to reason about, the client retries
				// or degrades gracefully, rather than rendering half a result.
				errResp := models.NewErrorResponse(
					requestID.(string),
					"ONNX_INFERENCE_ERROR",
					fmt.Sprintf("Inference failed on face index %d", i),
					map[string][]string{
						fmt.Sprintf("faces[%d]", i): {err.Error()},
					},
				)
				c.JSON(http.StatusInternalServerError, errResp)
				return
			}

			// Argmax over the 7-class probability distribution.
			// Identical logic to PredictHandler, linear scan beats sorting
			// for N=7.
			maxIndex := 0
			maxProb := float64(0.0)
			probabilityMap := make(map[string]float64)

			for k, prob := range probabilities {
				p := float64(prob)
				probabilityMap[EmotionLabels[k]] = p
				if p > maxProb {
					maxProb = p
					maxIndex = k
				}
			}

			results = append(results, BatchFaceResult{
				Index:           i,
				DominantEmotion: EmotionLabels[maxIndex],
				Confidence:      maxProb,
				Probabilities:   probabilityMap,
			})
		}

		inferenceMs := float64(time.Since(startTime).Microseconds()) / 1000.0

		// 6. Return all results under the standard envelope
		// inference_time_ms here covers the entire batch, not a single face.
		// If you need per-face timing for profiling, add a TimingMs field to
		// BatchFaceResult and record time.Since inside the loop — but don't
		// ship that in production unless you actually need it, it's noise.
		anyResults := make([]any, len(results))
		for i, r := range results {
			anyResults[i] = r
		}
		successResp := models.NewBatchSuccessResponse(requestID.(string), anyResults, inferenceMs, len(req.Faces))

		c.JSON(http.StatusOK, successResp)
	}
}
