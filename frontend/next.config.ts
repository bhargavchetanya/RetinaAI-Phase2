import type { NextConfig } from "next";

// Static export: `npm run build` writes plain HTML/JS to frontend/out, which server.py serves.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
