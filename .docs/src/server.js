"use strict";

const http = require("node:http");
const { URL } = require("node:url");

function json(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(payload, null, 2));
}

function text(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "content-type": "text/plain; charset=utf-8",
  });
  response.end(payload);
}

function notFound(response) {
  json(response, 404, {
    error: {
      code: "not_found",
      message: "The requested resource was not found.",
    },
  });
}

function badRequest(response, message) {
  json(response, 400, {
    error: {
      code: "bad_request",
      message,
    },
  });
}

async function readJson(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  }

  if (chunks.length === 0) {
    return {};
  }

  const raw = Buffer.concat(chunks).toString("utf8");
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error("Request body must be valid JSON.");
  }
}

function validateServicePayload(payload) {
  const requiredFields = ["name", "owner", "tier"];
  for (const field of requiredFields) {
    if (typeof payload[field] !== "string" || payload[field].trim() === "") {
      return `${field} is required and must be a non-empty string.`;
    }
  }
  return null;
}

function validateCheckPayload(payload) {
  const requiredFields = ["category", "status", "summary"];
  for (const field of requiredFields) {
    if (typeof payload[field] !== "string" || payload[field].trim() === "") {
      return `${field} is required and must be a non-empty string.`;
    }
  }

  if (!["pass", "warn", "fail"].includes(payload.status)) {
    return "status must be one of: pass, warn, fail.";
  }

  return null;
}

function createRequestHandler({ config, store }) {
  const startedAt = Date.now();
  let requestCount = 0;

  return async (request, response) => {
    requestCount += 1;
    const method = request.method || "GET";
    const requestUrl = new URL(request.url || "/", "http://127.0.0.1");
    const pathname = requestUrl.pathname;

    try {
      if (method === "GET" && pathname === "/health") {
        return json(response, 200, {
          status: "ok",
          service: config.serviceName,
          version: config.serviceVersion,
          uptimeSeconds: Math.round((Date.now() - startedAt) / 1000),
          timestamp: new Date().toISOString(),
        });
      }

      if (method === "GET" && pathname === "/ready") {
        return json(response, 200, {
          status: "ready",
          checks: {
            configLoaded: true,
            storeReachable: true,
          },
        });
      }

      if (method === "GET" && pathname === "/metrics") {
        return text(
          response,
          200,
          [
            "# HELP app_requests_total Total HTTP requests handled by the service",
            "# TYPE app_requests_total counter",
            `app_requests_total ${requestCount}`,
          ].join("\n"),
        );
      }

      if (method === "GET" && pathname === "/v1/services") {
        return json(response, 200, {
          data: store.listServices(),
        });
      }

      if (method === "POST" && pathname === "/v1/services") {
        const payload = await readJson(request);
        const validationError = validateServicePayload(payload);
        if (validationError) {
          return badRequest(response, validationError);
        }

        const service = store.createService(payload);
        return json(response, 201, {
          data: service,
        });
      }

      const serviceMatch = pathname.match(/^\/v1\/services\/([^/]+)$/);
      if (method === "GET" && serviceMatch) {
        const service = store.getService(serviceMatch[1]);
        if (!service) {
          return notFound(response);
        }

        return json(response, 200, {
          data: service,
        });
      }

      const checksMatch = pathname.match(/^\/v1\/services\/([^/]+)\/checks$/);
      if (checksMatch && method === "GET") {
        const service = store.getService(checksMatch[1]);
        if (!service) {
          return notFound(response);
        }

        return json(response, 200, {
          data: store.listChecks(checksMatch[1]),
        });
      }

      if (checksMatch && method === "POST") {
        const payload = await readJson(request);
        const validationError = validateCheckPayload(payload);
        if (validationError) {
          return badRequest(response, validationError);
        }

        const check = store.addCheck(checksMatch[1], payload);
        if (!check) {
          return notFound(response);
        }

        return json(response, 201, {
          data: check,
        });
      }

      return notFound(response);
    } catch (error) {
      return json(response, 500, {
        error: {
          code: "internal_error",
          message: error.message || "Unexpected server error.",
        },
      });
    }
  };
}

function createServer({ config, store }) {
  return http.createServer(createRequestHandler({ config, store }));
}

module.exports = {
  createRequestHandler,
  createServer,
};
