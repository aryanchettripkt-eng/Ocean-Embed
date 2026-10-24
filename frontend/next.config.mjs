/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    qualities: [75, 85],
  },
  experimental: {
    viewTransition: true,
    optimizePackageImports: ["lucide-react", "recharts"],
  },
};

export default nextConfig;
