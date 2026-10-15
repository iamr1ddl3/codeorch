// Sandbox config. Type-checking and linting run as their own steps in
// check.sh, so `next build` is told to skip them — each error is counted once,
// under the step that owns it.
/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: true },
  images: { unoptimized: true },
};

export default nextConfig;
