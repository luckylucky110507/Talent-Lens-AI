import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  root: ".",
  server: {
    host: "0.0.0.0",
    port: 5000,
    allowedHosts: true,
  },
  build: {
    outDir: "frontend/dist",
    emptyOutDir: true,
  },
});