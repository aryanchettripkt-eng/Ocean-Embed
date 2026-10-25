import { Pixelify_Sans, IBM_Plex_Mono } from "next/font/google";

export const pixelifySans = Pixelify_Sans({
  subsets: ["latin"],
  weight: ["600"],
  display: "swap",
  adjustFontFallback: true,
  variable: "--font-pixelify",
});

export const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
  adjustFontFallback: true,
  variable: "--font-mono",
});
