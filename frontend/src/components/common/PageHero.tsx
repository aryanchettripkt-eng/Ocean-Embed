"use client";

/**
 * Hero band for the content pages (About, Validation): the landing's underwater artwork and
 * Manrope display type, fading into the page, with a short GSAP entrance. Content cards below
 * overlap its lower edge, the way the Explorer's HUD floats over the map.
 */

import Image from "next/image";
import React, { useEffect, useRef } from "react";
import { manrope } from "../landing/fonts";
import { gsap, prefersReducedMotion } from "../../lib/motion";

interface PageHeroProps {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
  /** Small pill links or chips under the description. */
  actions?: React.ReactNode;
}

export function PageHero({ eyebrow, title, children, actions }: PageHeroProps) {
  const root = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = root.current;
    if (!el || prefersReducedMotion()) return;
    const ctx = gsap.context(() => {
      gsap.fromTo("[data-hero-art]", { scale: 1.08 }, { scale: 1, duration: 2.2, ease: "power2.out" });
      gsap.fromTo(
        "[data-hero-enter]",
        { y: 18, autoAlpha: 0 },
        { y: 0, autoAlpha: 1, duration: 0.9, ease: "power3.out", stagger: 0.09, clearProps: "transform,opacity,visibility" }
      );
    }, el);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={root} aria-labelledby="page-hero-title" className={`relative isolate overflow-hidden ${manrope.variable}`}>
      {/* Artwork: the landing's sunlit (light) and deep (dark) water, surface end up */}
      <div aria-hidden="true" className="absolute inset-0 -z-10">
        <div data-hero-art className="absolute inset-0">
          <Image
            unoptimized
            fill
            priority
            sizes="100vw"
            src="/landing/ocean/light-1600.webp"
            alt=""
            className="object-cover object-[50%_18%] dark:hidden"
          />
          <Image
            unoptimized
            fill
            priority
            sizes="100vw"
            src="/landing/ocean/dark-1600.webp"
            alt=""
            className="hidden object-cover object-[50%_18%] dark:block"
          />
        </div>
        {/* Legibility on the left, then fade into the page background */}
        <div className="absolute inset-0 bg-gradient-to-r from-[#062a2e]/70 via-[#062a2e]/30 to-transparent dark:from-black/60 dark:via-black/25" />
        <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-neutral-100 dark:to-[#0a0c0d]" />
      </div>

      <div className="mx-auto max-w-6xl px-4 pb-28 pt-14 sm:px-6 sm:pt-20 lg:px-8">
        <p
          data-hero-enter
          className="text-[11px] font-medium uppercase tracking-[0.18em] text-white/75"
        >
          {eyebrow}
        </p>
        <h1
          id="page-hero-title"
          data-hero-enter
          className="mt-3 max-w-3xl text-[clamp(32px,4.2vw,56px)] font-normal leading-[1.1] tracking-[-0.01em] text-white"
          style={{ fontFamily: "var(--font-manrope), sans-serif" }}
        >
          {title}
        </h1>
        {children && (
          <div data-hero-enter className="mt-4 max-w-2xl text-sm leading-relaxed text-white/80 sm:text-base">
            {children}
          </div>
        )}
        {actions && (
          <div data-hero-enter className="mt-6 flex flex-wrap gap-2">
            {actions}
          </div>
        )}
        {/* In flow under the actions on phones; in the corner (positioned on the section) from sm up */}
        <p className="mt-5 text-[10px] text-white/55 sm:absolute sm:bottom-28 sm:right-6 sm:mt-0">
          AI-generated illustration; not observational data.
        </p>
      </div>
    </section>
  );
}

/** Glass pill link for the hero's action row. */
export const heroActionClass =
  "inline-flex h-9 items-center gap-1.5 rounded-full border border-white/25 bg-white/10 px-4 text-xs font-medium text-white backdrop-blur-md transition-colors hover:bg-white/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white";
