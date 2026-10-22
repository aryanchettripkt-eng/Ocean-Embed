import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OceanScope AI — Subsurface Ocean Temperature Reconstruction",
  description: "AI-powered daily subsurface ocean temperature mapping in the North Indian Ocean",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen bg-slate-950 text-slate-100">
        {children}
      </body>
    </html>
  );
}
