import { defineConfig } from "vite";

const apiTarget = process.env.VITE_PROXY_TARGET || "https://tox-agent.web.app";

const proxyRoutes = ["/health", "/predict", "/predict/batch", "/explain", "/analyze"];
const proxy = Object.fromEntries(
  proxyRoutes.map((route) => [
    route,
    {
      target: apiTarget,
      changeOrigin: true,
      secure: true,
    },
  ]),
);

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy,
  },
});
