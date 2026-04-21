"use strict";

const { randomUUID } = require("node:crypto");

function isoNow() {
  return new Date().toISOString();
}

function createService(input) {
  const now = isoNow();
  return {
    id: randomUUID(),
    name: input.name,
    owner: input.owner,
    tier: input.tier,
    description: input.description || "",
    lifecycle: input.lifecycle || "active",
    createdAt: now,
    updatedAt: now,
  };
}

function createCheck(serviceId, input) {
  const now = isoNow();
  return {
    id: randomUUID(),
    serviceId,
    category: input.category,
    status: input.status,
    summary: input.summary,
    evidenceUrl: input.evidenceUrl || "",
    checkedAt: input.checkedAt || now,
    createdAt: now,
  };
}

function createSeedData() {
  const service = createService({
    name: "tox-agent-web",
    owner: "platform@toxagent.local",
    tier: "tier-1",
    description: "Primary customer-facing application for running toxicity analyses.",
    lifecycle: "active",
  });

  const checks = [
    createCheck(service.id, {
      category: "security",
      status: "pass",
      summary: "Threat model reviewed and no high-severity gaps are open.",
      evidenceUrl: "https://internal.example/security-review",
    }),
    createCheck(service.id, {
      category: "operability",
      status: "pass",
      summary: "Synthetic health checks and log-based alerts are configured.",
      evidenceUrl: "https://internal.example/alerts",
    }),
  ];

  return {
    services: [service],
    checks,
  };
}

function createStore(seed = createSeedData()) {
  const state = {
    services: [...seed.services],
    checks: [...seed.checks],
  };

  return {
    listServices() {
      return state.services.map((service) => ({
        ...service,
        checks: state.checks.filter((check) => check.serviceId === service.id),
      }));
    },

    createService(input) {
      const service = createService(input);
      state.services.push(service);
      return service;
    },

    getService(serviceId) {
      const service = state.services.find((entry) => entry.id === serviceId);
      if (!service) {
        return null;
      }

      return {
        ...service,
        checks: state.checks.filter((check) => check.serviceId === service.id),
      };
    },

    addCheck(serviceId, input) {
      const service = state.services.find((entry) => entry.id === serviceId);
      if (!service) {
        return null;
      }

      service.updatedAt = isoNow();
      const check = createCheck(serviceId, input);
      state.checks.push(check);
      return check;
    },

    listChecks(serviceId) {
      return state.checks.filter((check) => check.serviceId === serviceId);
    },
  };
}

module.exports = {
  createStore,
};
