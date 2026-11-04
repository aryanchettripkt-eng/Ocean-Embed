"use client";
import Image from "next/image";
import React, { useEffect, useRef, useState } from "react";
import { fishLayout } from "./fish-layout";
import styles from "./landing.module.css";
import type { OceanScene } from "./ocean-scene";

export function HeroScene({
  paused,
  theme,
}: {
  paused: boolean;
  theme: "light" | "dark";
}) {
  const host = useRef<HTMLDivElement>(null);
  const controller = useRef<OceanScene | null>(null);
  const latest = useRef({ paused, theme });
  latest.current = { paused, theme };
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let disposed = false;
    let generation = 0;
    let pending: AbortController | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const reset = () => {
      generation++;
      pending?.abort();
      controller.current?.dispose();
      controller.current = null;
      setReady(false);
      clearTimeout(timer);
      if (reduced.matches) return;
      const current = generation;
      const aborter = new AbortController();
      pending = aborter;
      // The poster paints first. Canvas code stays outside the initial landing bundle.
      timer = setTimeout(() => {
        import("./ocean-scene")
          .then(async ({ createOceanScene }) => {
            if (disposed || current !== generation) return;
            const scene = await createOceanScene(
              element,
              latest.current.theme,
              () => {
                if (!disposed && current === generation) setReady(false);
              },
              aborter.signal,
            );
            if (disposed || current !== generation) {
              scene?.dispose();
              return;
            }
            controller.current = scene;
            if (scene) {
              scene.setPaused(latest.current.paused);
              scene.setTheme(latest.current.theme);
              setReady(true);
            }
          })
          .catch(() => {
            if (!disposed) setReady(false);
          });
      }, 180);
    };
    reset();
    reduced.addEventListener("change", reset);
    return () => {
      disposed = true;
      generation++;
      pending?.abort();
      clearTimeout(timer);
      reduced.removeEventListener("change", reset);
      controller.current?.dispose();
      controller.current = null;
    };
  }, []);
  useEffect(() => {
    controller.current?.setPaused(paused);
  }, [paused]);
  useEffect(() => {
    controller.current?.setTheme(theme);
  }, [theme]);
  return (
    <div className={styles.sceneViewport} aria-hidden="true">
      <div className={styles.stage} ref={host} data-scene-ready={ready}>
        {/* Native responsive images are also the no-JS / renderer-failure fallback. */}
        <picture>
          <source
            media="(max-width: 767px)"
            srcSet="/landing/ocean/light-1024.webp"
          />
          <source
            srcSet="/landing/ocean/light-640.webp 640w, /landing/ocean/light-1024.webp 1024w, /landing/ocean/light-1600.webp 1600w"
            sizes="100vw"
          />
          <Image
            unoptimized
            className={`${styles.sceneImage} ${styles.lightPoster}`}
            src="/landing/ocean/light-1600.webp"
            alt=""
            width="1672"
            height="941"
            fetchPriority="high"
          />
        </picture>
        <picture>
          <source
            media="(max-width: 767px)"
            srcSet="/landing/ocean/dark-1024.webp"
          />
          <source
            srcSet="/landing/ocean/dark-640.webp 640w, /landing/ocean/dark-1024.webp 1024w, /landing/ocean/dark-1600.webp 1600w"
            sizes="100vw"
          />
          <Image
            unoptimized
            className={`${styles.sceneImage} ${styles.darkPoster}`}
            src="/landing/ocean/dark-1600.webp"
            alt=""
            width="1672"
            height="941"
            fetchPriority="high"
          />
        </picture>
        <div className={styles.staticFish}>
          {fishLayout.map((fish, i) => (
            <Image
              unoptimized
              key={i}
              src="/landing/ocean/fish.webp"
              alt=""
              width="384"
              height="128"
              style={{
                left: `${fish.x * 100}%`,
                top: `${fish.y * 100}%`,
                width: `${fish.width * 100}%`,
                opacity: fish.opacity,
              }}
            />
          ))}
        </div>
      </div>
      <div className={styles.sceneVignette} />
    </div>
  );
}
