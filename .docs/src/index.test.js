"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { Readable } = require("node:stream");

const { createRequestHandler } = require("./server");
const { createStore } = require("./store");

async function invoke({ method, url, body, store = createStore() }) {
  const handler = createRequestHandler({
    config: {
      serviceName: "test-service",
      serviceVersion: "test",
    },
    store,
  });

  const request = Readable.from(body ? [JSON.stringify(body)] : []);
  request.method = method;
  request.url = url;

  let responseBody = "";
  const response = {
    statusCode: 200,
    headers: {},
    writeHead(statusCode, headers) {
      this.statusCode = statusCode;
      this.headers = headers;
    },
    end(chunk = "") {
      responseBody += chunk;
    },
  };

  await handler(request, response);
  return {
    statusCode: response.statusCode,
    headers: response.headers,
    json: responseBody ? JSON.parse(responseBody) : null,
    rawBody: responseBody,
    store,
  };
}

test("GET /health returns service metadata", async () => {
  const response = await invoke({
    method: "GET",
    url: "/health",
  });

  assert.equal(response.statusCode, 200);
  assert.equal(response.json.status, "ok");
  assert.equal(response.json.service, "test-service");
});

test("POST /v1/services creates a service", async () => {
  const response = await invoke({
    method: "POST",
    url: "/v1/services",
    body: {
      name: "sample-service",
      owner: "owner@example.com",
      tier: "tier-2",
      description: "Test service",
    },
  });

  assert.equal(response.statusCode, 201);
  assert.equal(response.json.data.name, "sample-service");
  assert.equal(response.json.data.owner, "owner@example.com");
});

test("POST /v1/services/:id/checks validates status", async () => {
  const store = createStore();
  const seedServiceId = store.listServices()[0].id;

  const response = await invoke({
    method: "POST",
    url: `/v1/services/${seedServiceId}/checks`,
    body: {
      category: "security",
      status: "broken",
      summary: "invalid status",
    },
    store,
  });

  assert.equal(response.statusCode, 400);
  assert.match(response.json.error.message, /status must be one of/);
});
