import type { Metadata } from "next";
import { Landing } from "../../components/landing/Landing";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "https://ocean-embed.vercel.app"),
  title: "OceanScope AI — Beneath the Surface",
  description:
    "AI-powered 3D subsurface ocean temperature reconstruction for the North Indian Ocean from surface dynamics and in-situ observations.",
  openGraph: {
    title: "OceanScope AI — Beneath the Surface",
    description:
      "AI-powered 3D subsurface ocean temperature reconstruction for the North Indian Ocean.",
    url: "/",
    siteName: "OceanScope AI",
    images: [
      {
        url: "/landing/og.png",
        width: 1200,
        height: 630,
        alt: "OceanScope AI — 3D Subsurface Ocean Temperature Reconstruction",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "OceanScope AI — Beneath the Surface",
    description:
      "AI-powered 3D subsurface ocean temperature reconstruction for the North Indian Ocean.",
    images: ["/landing/og.png"],
  },
};

export default function LandingPage() {
  return <Landing />;
}
