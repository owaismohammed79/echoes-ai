import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

/**
 * Hostnames the dev server may answer to beyond the defaults.
 *
 * `localhost`, every `*.localhost` name and every IP address are allowed by Vite
 * already, so this list is only for a host that reaches the dev server from
 * somewhere else: a tunnel, or a preview deployment proxied onto it.
 *
 * Vercel is allowed as a *suffix* (a leading dot covers the domain and all of
 * its subdomains) rather than as a named deployment, because a preview URL is
 * generated per commit and a hardcoded one is stale by the next push. Anything
 * else comes from `VITE_ALLOWED_HOST`, so a tunnel is configuration rather than
 * a URL committed to the repository. Vite also honours its own
 * `__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS` if you would rather use that.
 *
 * A deployed client never runs this server -- Vercel serves the static build --
 * so this is a development and preview concern only. `preview.allowedHosts`
 * defaults to this list.
 */
const DEFAULT_ALLOWED_HOSTS = [".vercel.app"];

export default defineConfig(({ mode }) => {
  // "" loads every variable, not only `VITE_*`: the host list is a server-side
  // setting, and Vite exposes `VITE_*` to client code alone. `loadEnv` rather
  // than `process.env`, so a value in `client/.env` behaves like one passed on
  // the command line.
  const env = loadEnv(mode, process.cwd(), "");
  const extraHosts = (env.VITE_ALLOWED_HOST ?? "")
    .split(",")
    .map((host) => host.trim())
    .filter(Boolean);

  return {
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
      allowedHosts: [...DEFAULT_ALLOWED_HOSTS, ...extraHosts],
      strictPort: true,
      // The FastAPI service runs beside the client in development. Proxying keeps
      // the engine's relative URLs (real-engine.ts) working locally without a
      // build-time VITE_API_BASE_URL, which is only needed cross-origin in prod.
      proxy: {
        "/api": "http://127.0.0.1:8000",
        "/static": "http://127.0.0.1:8000",
      },
    },
  };
});
