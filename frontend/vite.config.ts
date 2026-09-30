import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND = process.env.VITE_BACKEND_ORIGIN ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // Proxying keeps the browser on a single origin, so the API's CORS allowlist
    // is never exercised during development.
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
