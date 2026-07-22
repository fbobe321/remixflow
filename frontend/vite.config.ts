import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build straight into the Python package's static dir so `pip install remixflow`
// ships the compiled UI (no Node needed by end users). Dev proxies /api to the
// FastAPI server on :8000.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../backend/remixflow/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8770",
    },
  },
});
