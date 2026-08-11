import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // dev: proxy API calls to the FastAPI app container
    proxy: {
      "/api": { target: "http://web:8800", changeOrigin: true },
    },
  },
});
