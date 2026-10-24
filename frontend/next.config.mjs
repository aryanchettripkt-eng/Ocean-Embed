/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    viewTransition: true,
    optimizePackageImports: ["lucide-react", "recharts"],
  },
};

export default nextConfig;
