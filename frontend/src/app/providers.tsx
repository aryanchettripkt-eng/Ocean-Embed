"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { usePathname } from "next/navigation";
import React, { useEffect, useRef, useState } from "react";
import { ArrivalOverlay } from "../components/transition/ArrivalOverlay";

export function Providers({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const focusExplorer = useRef(false);
  useEffect(() => {
    const requestFocus = () => { focusExplorer.current = true; };
    window.addEventListener("oceanscope:focus-explorer", requestFocus);
    return () => window.removeEventListener("oceanscope:focus-explorer", requestFocus);
  }, []);
  useEffect(() => {
    if (pathname !== "/explore" || !focusExplorer.current) return;
    // The Explorer swaps its first-render markup for the real page, replacing the heading node, so
    // keep watching briefly after the first focus: if focus drops back to <body>, focus the new
    // heading. Stop as soon as focus is somewhere else on purpose (the user tabbed or clicked).
    let settle: ReturnType<typeof setTimeout> | undefined;
    const focus = () => {
      const heading = document.getElementById("explorer-heading");
      if (!heading) return;
      const active = document.activeElement;
      if (active === heading) return;
      if (settle && active && active !== document.body) {
        observer.disconnect();
        return;
      }
      heading.focus();
      focusExplorer.current = false;
      settle ??= setTimeout(() => observer.disconnect(), 3000);
    };
    const observer = new MutationObserver(focus);
    observer.observe(document.body, { childList: true, subtree: true });
    focus();
    const timer = setTimeout(() => observer.disconnect(), 10000);
    return () => { observer.disconnect(); clearTimeout(timer); clearTimeout(settle); };
  }, [pathname]);
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: Infinity,
            refetchOnWindowFocus: false,
            retry: 2,
          },
        },
      })
  );

  const isMocking = process.env.NEXT_PUBLIC_USE_MOCKS === "true";
  const [mockReady, setMockReady] = useState(() => !isMocking);

  useEffect(() => {
    if (isMocking) {
      import("../mocks/browser")
        .then(async ({ worker }) => {
          if (worker) {
            await worker.start({
              onUnhandledRequest: "bypass",
              serviceWorker: {
                url: "/mockServiceWorker.js",
              },
            });
          }
          setMockReady(true);
        })
        .catch((err) => {
          console.error("Failed to start MSW worker:", err);
          setMockReady(true);
        });
    } else {
      setMockReady(true);
    }
  }, [isMocking]);

  const isMarketingRoute = pathname === "/" || pathname === "/welcome";

  if (isMocking && !mockReady && !isMarketingRoute) {
    return (
      <>
        <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-400">
          <div className="flex items-center space-x-3">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-500 border-t-transparent" />
            <span>Initializing mock service...</span>
          </div>
        </div>
        <ArrivalOverlay />
      </>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* Landing → Explorer transition layer; lives here so it survives the route change */}
      <ArrivalOverlay />
    </QueryClientProvider>
  );
}
