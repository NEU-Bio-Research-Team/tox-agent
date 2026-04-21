"use strict";

const { loadConfig } = require("./config");
const { createServer } = require("./server");
const { createStore } = require("./store");

function start() {
  const config = loadConfig();
  const store = createStore();
  const server = createServer({ config, store });

  server.listen(config.port, config.host, () => {
    process.stdout.write(
      `${config.serviceName} listening on http://${config.host}:${config.port}\n`,
    );
  });

  return server;
}

if (require.main === module) {
  start();
}

module.exports = {
  start,
};
