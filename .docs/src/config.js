"use strict";

const DEFAULT_PORT = 3000;
const DEFAULT_HOST = "0.0.0.0";
const DEFAULT_NODE_ENV = "development";
const DEFAULT_SERVICE_NAME = "readiness-control-api";
const DEFAULT_VERSION = "1.0.0";

function parsePort(rawValue) {
  const parsed = Number(rawValue);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) {
    throw new Error("PORT must be an integer between 1 and 65535.");
  }
  return parsed;
}

function loadConfig(env = process.env) {
  return {
    host: env.HOST || DEFAULT_HOST,
    port: env.PORT ? parsePort(env.PORT) : DEFAULT_PORT,
    nodeEnv: env.NODE_ENV || DEFAULT_NODE_ENV,
    serviceName: env.SERVICE_NAME || DEFAULT_SERVICE_NAME,
    serviceVersion: env.SERVICE_VERSION || DEFAULT_VERSION,
  };
}

module.exports = {
  loadConfig,
};
