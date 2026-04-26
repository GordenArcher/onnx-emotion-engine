package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/handlers"
	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/middleware"
	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/ml"
	"github.com/GordenArcher/onnx-emotion-engine/backend/internal/models"
	"github.com/gin-gonic/gin"
)

func main() {
	// Gin's default mode is "debug," which is useful during development because
	// it logs every route hit and includes a panic-recovery page with stack
	// traces. In production, those logs are pure noise—they drown out
	// application-level logging and add latency to every request. Release mode
	// strips all of that, keeping stdout clean for our own structured output.
	gin.SetMode(gin.ReleaseMode)

	fmt.Println("Starting ONNX Emotion Engine Backend...")

	// Model Initialization
	// Loading an ONNX model is a relatively expensive operation: the runtime
	// parses the protobuf graph, allocates tensor memory, and verifies that the
	// available hardware (CPU, potentially GPU) can execute every operator.
	// We do this synchronously at startup so that by the time the HTTP server
	// begins accepting traffic, the model is already warm and ready to serve
	// predictions with zero cold-start latency.
	model, err := ml.NewEmotionModel("emotion_model.onnx")
	if err != nil {
		// A missing or corrupted model file is a hard startup failure. No amount
		// of retry logic or graceful degradation can salvage a prediction service
		// that cannot make predictions. We use log.Fatalf to print the error and
		// immediately exit with a non-zero status code, which signals to our
		// orchestrator (Docker Compose, Kubernetes) that this container is
		// unhealthy and should not receive traffic.
		log.Fatalf("FATAL: Failed to initialize ML model: %v", err)
	}
	fmt.Println("ML Model loaded successfully into memory.")

	// ONNX Runtime allocates its own memory pools outside Go's garbage-collected
	// heap. If we don't explicitly release those resources before the process
	// exits, we are relying on the OS to clean up after us—which works, but
	// leaks memory in long-running tests or when the model is reloaded for
	// hot-swapping. The deferred Destroy() call guarantees that every byte
	// allocated by the runtime is returned to the OS, maintaining a flat memory
	// profile across the server's lifetime.
	defer model.Destroy()

	// Router Setup
	// gin.New() gives us a completely blank router with no middleware attached.
	// This is preferable to gin.Default() because Default() includes gin.Logger(),
	// which would duplicate our structured request logging and write to stdout
	// in a format we cannot control. Starting from scratch lets us compose
	// exactly the middleware stack we need, nothing more.
	router := gin.New()

	// gin.Recovery() is the safety net that catches panics in any handler
	// goroutine and converts them into 500 Internal Server Error responses.
	// Without it, a single nil-pointer dereference in the prediction handler
	// would crash the entire server, taking down every concurrent request.
	// Recovery ensures that one bad request can only poison itself, not the
	// whole process.
	router.Use(gin.Recovery())

	// RequestID injects a unique UUID into every request's context. This
	// identifier propagates through our logs, the prediction pipeline, and
	// the HTTP response headers, forming a trace that connects a client-side
	// error to a specific server-side log line. In a production incident,
	// tracing a request ID cuts mean-time-to-resolution from hours to minutes.
	router.Use(middleware.RequestID())

	origins := os.Getenv("ALLOWED_ORIGINS")
	allowedOrigins := []string{"http://localhost:5173"}
	if origins != "" {
		allowedOrigins = append(allowedOrigins, strings.Split(origins, ",")...)
	}
	router.Use(middleware.CORS(allowedOrigins))

	// Root Endpoint — Built-in API Documentation
	// A landing page at "/" transforms the server from a black box into a self-
	// documenting service. Anyone who hits the root URL—whether a frontend
	// engineer integrating for the first time, a DevOps engineer debugging a
	// health check, or a product manager exploring the API—gets a complete map
	// of every available route, its expected input, its output shape, and a
	// working curl command they can copy-paste into a terminal.
	//
	// This documentation is always in sync with the code because it lives in
	// the same binary. There is no separate wiki or Postman collection to
	// maintain and inevitably forget to update.
	//
	// We return our standard APIResponse envelope so the root endpoint follows
	// the same contract as every other endpoint. Clients can reuse their
	// response-parsing logic without special-casing the documentation route.
	router.GET("/", func(c *gin.Context) {
		requestID, _ := c.Get("request_id")

		docs := models.APIDocumentation{
			Service:     "ONNX Emotion Engine",
			Version:     "1.0.0",
			Description: "Real-time facial emotion inference powered by MobileNetV2 running inside the ONNX Runtime. Accepts a flat array of 6,912 normalized pixels (48×48 grayscale expanded to 3 channels) and returns a probability distribution over 7 emotion classes.",
			Model: models.ModelInfo{
				Architecture:     "MobileNetV2 (transfer learning from ImageNet)",
				InputShape:       "[1, 3, 48, 48] — batch=1, channels=3 (grayscale replicated), height=48, width=48",
				InputLength:      6912,
				PixelRange:       "[0.0, 1.0] — pre-normalized with ImageNet mean/std by the Python export script",
				OutputShape:      "[1, 7] — 7-class probability distribution",
				EmotionClasses:   []string{"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"},
				Framework:        "ONNX Runtime",
				InferenceLatency: "~3–8ms on modern CPU",
			},
			Routes: []models.RouteDoc{
				{
					Method:      "GET",
					Path:        "/",
					Description: "This documentation page. Lists all available endpoints with their request/response contracts.",
					Example:     "curl http://localhost:8080/",
				},
				{
					Method:      "GET",
					Path:        "/api/v1/health",
					Description: "Health check endpoint for load balancers, container orchestrators (Kubernetes liveness/readiness probes), and manual smoke tests.",
					Response: map[string]any{
						"status": "200 OK",
						"body": map[string]string{
							"status": "healthy",
						},
					},
					Example: "curl http://localhost:8080/api/v1/health",
				},
				{
					Method:      "POST",
					Path:        "/api/v1/predict",
					Description: "Runs facial emotion inference on the supplied pixel array. The request must carry a JSON body with a single key \"pixels\" whose value is an array of exactly 6,912 float64 numbers.",
					Headers: map[string]string{
						"Content-Type": "application/json",
					},
					Request: map[string]string{
						"pixels": "[float64 array, length 6912] — 48×48 grayscale image expanded to 3 channels, normalized to ImageNet statistics",
					},
					ResponseSuccess: models.APIResponse{
						Status:    "success",
						Message:   "Emotion inferred successfully",
						Code:      "EMOTION_INFERRED",
						RequestID: "<uuid-string>",
						Data: models.EmotionData{
							DominantEmotion: "Happy",
							Confidence:      0.9521,
							Probabilities: map[string]float64{
								"Angry":    0.0012,
								"Disgust":  0.0003,
								"Fear":     0.0021,
								"Happy":    0.9521,
								"Neutral":  0.0310,
								"Sad":      0.0087,
								"Surprise": 0.0046,
							},
						},
						Metadata: models.Metadata{
							InferenceTimeMs: 4.23,
							ModelVersion:    "mobilenet-v2-v1.0.0",
							Timestamp:       "2026-04-25T12:34:56Z",
						},
					},
					ResponseError: models.APIResponse{
						Status:    "error",
						Message:   "Pixel array does not match required model input shape",
						Code:      "INVALID_PIXEL_DIMENSIONS",
						RequestID: "<uuid-string>",
						Errors: map[string][]string{
							"pixels": {"Expected array of length 6912 (48x48x3), received 500"},
						},
						Metadata: models.Metadata{
							Timestamp: "2026-04-25T12:34:56Z",
						},
					},
					Example: `curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0.5, 0.5, 0.5, ...]}'  # 6,912 floats total`,
				},
			},
			Notes: []string{
				"The pixel array must be exactly 6,912 elements long. Shorter or longer arrays will be rejected with a 400 error and a descriptive message.",
				"All responses carry a request_id (UUID v4) echoed in the X-Request-ID header and the JSON body. Quote this ID when reporting bugs.",
				"Probabilities sum to approximately 1.0 (subject to floating-point rounding). The dominant_emotion is the argmax of the probability distribution.",
				"This API is versioned under /api/v1. Breaking changes will be released under /api/v2, leaving /api/v1 intact for existing clients.",
			},
		}

		// Wrap the documentation data in our standard APIResponse envelope
		response := models.NewSuccessResponse(
			requestID.(string),
			docs,
			0.0, // No inference occurred for this documentation endpoint
		)

		c.JSON(http.StatusOK, response)
	})

	// API Routes
	// Versioning the API under /api/v1 costs us nothing today but buys us
	// invaluable flexibility tomorrow. If we ever change the request schema
	// (e.g., switching from base64-encoded images to raw binary uploads),
	// we can deploy the new endpoint at /api/v2 without breaking existing
	// mobile clients that still speak v1.
	v1 := router.Group("/api/v1")
	{
		// Dependency injection via closure: the handler captures the model
		// pointer from the outer scope rather than reaching into a global
		// variable. This makes the handler testable—we can inject a mock
		// model during unit tests without touching the real ONNX runtime.
		v1.POST("/predict", handlers.PredictHandler(model))

		// The /health endpoint serves two distinct audiences. Load balancers
		// and container orchestrators poll it every few seconds to decide
		// whether traffic should be routed to this instance. Human operators
		// curl it during incidents to quickly rule out "is the port even open?"
		//
		// We wrap the response in our standard APIResponse envelope. This lets
		// monitoring tools parse health-check responses with the same code they
		// use for every other endpoint, and the request_id gives operators an
		// immediate trace anchor if they correlate a health-check failure with
		// other log lines from the same time window.
		v1.GET("/health", func(c *gin.Context) {
			requestID, _ := c.Get("request_id")

			response := models.NewSuccessResponse(
				requestID.(string),
				gin.H{"status": "healthy"},
				0.0,
			)

			c.JSON(http.StatusOK, response)
		})
	}

	v2 := router.Group("/api/v2")
	{
		v2.POST("/predict/batch", handlers.BatchPredictHandler(model))
	}

	// HTTP Server Configuration
	// We explicitly construct the http.Server struct rather than calling
	// router.Run() because we need a handle on the server to gracefully shut it
	// down later. The Addr and Handler fields are the minimum configuration,
	// but in a production deployment you would also set ReadTimeout and
	// WriteTimeout to prevent slow clients from exhausting connection pools.
	port := ":8080"
	srv := &http.Server{
		Addr:    port,
		Handler: router,
	}

	// Concurrent Server Startup
	// ListenAndServe is a blocking call. If we ran it on the main goroutine,
	// we would never reach the graceful-shutdown logic below. Launching it in
	// a separate goroutine decouples the server's lifecycle from the main
	// function, allowing us to simultaneously serve traffic and wait for
	// termination signals.
	go func() {
		fmt.Printf("Server is listening on http://localhost%s\n", port)
		fmt.Printf("Visit http://localhost%s/ for complete API documentation.\n", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			// http.ErrServerClosed is the expected error when we explicitly call
			// srv.Shutdown(). Any other error (e.g., port 8080 already bound)
			// is fatal and should tear down the entire process.
			log.Fatalf("FATAL: Server failed to start: %v", err)
		}
	}()

	// Graceful Shutdown
	// A naive server that exits on SIGINT abandons every in-flight request.
	// Clients see connection resets, and partially-processed predictions are
	// lost. This channel-based approach intercepts the termination signal,
	// gives the server a grace period to drain existing connections, and only
	// then exits—turning a chaotic crash into an orderly shutdown.
	//
	// SIGINT  → generated by Ctrl+C in a terminal
	// SIGTERM → sent by Kubernetes when a pod is being evicted or scaled down
	// Both signals demand a clean exit, so we handle them identically.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	fmt.Println("\nShutting down server gracefully...")

	// The context with a 5-second timeout serves as a deadline. srv.Shutdown()
	// will wait for all active requests to finish, but if any request takes
	// longer than 5 seconds, Shutdown returns immediately and the server exits
	// anyway. Five seconds is a pragmatic balance: long enough for a typical
	// prediction request (which completes in <100ms) to finish, but short
	// enough that a stalled handler cannot delay the shutdown indefinitely.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		// If Shutdown returns an error, it means the deadline expired before
		// all connections drained. We log the event and exit, trusting that
		// clients have retry logic. In the future, this could be wired into
		// an alerting system to track how often we force-close connections.
		log.Fatal("Server forced to shutdown:", err)
	}

	fmt.Println("Server exited cleanly.")
}
