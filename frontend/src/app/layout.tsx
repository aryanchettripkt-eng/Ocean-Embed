import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Banner } from "../components/common/Banner";
import { Header } from "../components/common/Header";

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
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
        <Providers>
          <Banner type="synthetic" />
          <Header />
          <div className="flex-1 flex flex-col">{children}</div>
        </Providers>
      </body>
    </html>
  );
}
