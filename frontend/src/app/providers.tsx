"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { useEffect, useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
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

  if (isMocking && !mockReady) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-400">
        <div className="flex items-center space-x-3">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-500 border-t-transparent" />
          <span>Initializing mock service...</span>
        </div>
      </div>
    );
  }

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
