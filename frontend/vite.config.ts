/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: the UI runs on :5173 and forwards /v1 and /health to the API on :8000,
// so the browser sees one origin (no CORS) just like in production, where
// FastAPI serves the built files itself.
const API = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: API, changeOrigin: true },
      "/health": { target: API, changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});