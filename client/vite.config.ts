import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    // The Vite dev server is what gets shown in the Preview panel; keep it on a
    // host the browser can reach and a port that will not collide with the API.
    host: "127.0.0.1",
    port: 5311,
    allowedHosts: "https://egotism-subprime-viable.ngrok-free.dev",
    strictPort: true,
    // The FastAPI service runs beside the client in development. Proxying keeps
    // the engine's relative URLs (real-engine.ts) working locally without a
    // build-time VITE_API_BASE_URL, which is only needed cross-origin in prod.
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/static": "http://127.0.0.1:8000",
    },
  },
});
