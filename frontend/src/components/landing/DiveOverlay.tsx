"use client";

/**
 * Landing half of the landing → Explorer arrival (src/lib/arrival.ts): on the CTA's
 * "oceanscope:dive" event, start loading the Explorer, lift the hero copy away and sink the
 * underwater scene while ArrivalOverlay sweeps the water surface down the screen — we rise out
 * of the sea. When the screen is covered, navigate; the overlay carries over to the Explorer.
 */
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../../lib/api/client";
import { beginArrival } from "../../lib/arrival";

export function DiveOverlay() {
  const router = useRouter();
  const queryClient = useQueryClient();

  useEffect(() => {
    const dive = (event: Event) => {
      const href = (event as CustomEvent<{ href: string }>).detail?.href ?? "/explore?view=3d";

      // The transition doubles as load time: fetch the route, map library and metadata now
      router.prefetch(href);
      import("maplibre-gl").catch(() => {});
      void queryClient.prefetchQuery({ queryKey: ["meta"], queryFn: () => apiClient.getMeta(), staleTime: Infinity });

      if (!beginArrival(() => router.push(href))) return;

      const landing = document.querySelector<HTMLElement>("[data-landing]");
      if (!landing) return;
      const exits = landing.querySelectorAll("[data-landing-exit]");
      const scene = landing.querySelector("[data-landing-scene]");
      // GSAP stays out of the initial landing bundle (DECISIONS 2026-11-05 budgets); the landing
      // motion has usually loaded it already, so this resolves immediately.
      void import("gsap").then(({ gsap }) => {
        gsap.to(exits, { y: -28, autoAlpha: 0, duration: 0.45, ease: "power2.in", stagger: 0.05 });
        if (scene) gsap.to(scene, { yPercent: 16, scale: 1.06, duration: 0.95, ease: "power2.in" });
      });
    };
    window.addEventListener("oceanscope:dive", dive);
    return () => window.removeEventListener("oceanscope:dive", dive);
  }, [router, queryClient]);

  return null;
}
