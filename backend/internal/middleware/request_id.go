package middleware

import (
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// RequestID returns a Gin middleware that stamps every HTTP request with a
// universally-unique identifier. In a monolithic application with synchronous
// request handling, this middleware might feel like overkill. In a distributed
// system—even one as small as a React frontend and a Go backend—it is the
// difference between a five-minute incident investigation and a five-hour one.
//
// The lifecycle of a request ID:
//  1. Generated here, before any handler logic runs.
//  2. Attached to the Gin context so every downstream handler and logging
//     call can access it without threading it through function parameters.
//  3. Reflected back to the client in the X-Request-ID response header so
//     the browser's Network tab shows the exact UUID. A user reporting a bug
//     can screenshot that header, and an engineer can grep the server logs
//     for that UUID to reconstruct the full request journey.
//  4. Embedded in every JSON response body (both success and error) by the
//     handler layer, closing the observability loop.
//
// We use UUID v4 (random) rather than a monotonically-increasing integer
// because randomness eliminates the coordination overhead of a central counter
// and makes request IDs globally unique across multiple replicas of this
// service without any shared state.
func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		// uuid.New() reads from crypto/rand under the hood, giving us
		// collision-resistant identifiers suitable for security-sensitive
		// contexts. The performance cost is roughly 1-2 microseconds—three
		// orders of magnitude smaller than the typical ONNX inference time
		// that dominates our request latency.
		requestID := uuid.New().String()

		// Gin's context is a key-value store scoped to a single HTTP request.
		// Values placed here are automatically garbage-collected when the
		// request completes, so we never leak memory even under sustained
		// high throughput. The "request_id" key is an implicit contract
		// between this middleware and every handler that calls c.Get("request_id").
		c.Set("request_id", requestID)

		// Setting the header on the response, not the request, is intentional.
		// The client sends us a request; we enrich it with an ID and echo that
		// ID back. If the client already sent an X-Request-ID header (e.g.,
		// from an upstream proxy or a mobile SDK), a production-hardened
		// version of this middleware would check for that value first and
		// propagate it rather than overwriting it. For now, generating our own
		// guarantees exactly one authoritative ID per request.
		c.Header("X-Request-ID", requestID)

		// c.Next() transfers control to the next middleware in the chain,
		// and eventually to the route handler. Execution resumes here after
		// the handler returns, which would be the place to add teardown logic
		// such as recording request duration metrics or finalizing a logging
		// span. Currently we have no post-processing, but the structural hook
		// is already in place.
		c.Next()
	}
}
