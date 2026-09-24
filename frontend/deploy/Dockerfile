# ToxAgent frontend: static SPA served by nginx.
#
# Build: docker build -f frontend/deploy/Dockerfile -t toxagent-frontend frontend
#
# Build context is frontend/ itself. The API base URL is not baked in here —
# the app reads it at runtime from same-origin relative paths (vite.config.ts
# dev proxy mirrors this); a reverse proxy or the deploy topology in front of
# this container is what actually routes /v1 and /health to the control
# plane. Nothing about that is this image's concern.
# Pinned by digest (2026-09-06) — W6-11, same reasoning as
# toxagent-control/deploy/Dockerfile.
FROM node:22-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10
COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
