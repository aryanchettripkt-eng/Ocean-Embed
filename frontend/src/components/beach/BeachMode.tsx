"use client";

/**
 * Beach mode (PRD_FRONTEND §3.4f, phones). One "Point at the sea" button: on tap, ask for
 * location (and on iOS the compass), call /nearest-ocean, then show that cell's /profile as
 * a big vertical temperature strip with distance and direction. Location is sent only on
 * tap, kept only in memory, and never stored in the browser.
 */

import React, { useCallback, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Compass, Loader2, MapPin, MessageCircle } from "lucide-react";

import { ApiError, apiClient, type MetaResponse, type NearestOceanResponse } from "../../lib/api/client";
import { useProfile } from "../../lib/api/hooks";
import { getColorHex } from "../../lib/colormaps";
import { formatCoord, formatDate, formatDepth, formatTemp } from "../../lib/format";
import { findClosestDepth } from "../../lib/url-state";

const GuideDrawer = dynamic(() => import("../guide/GuideDrawer").then((m) => m.GuideDrawer), { ssr: false });

const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];

export function compassWord(bearingDeg: number): string {
  const i = Math.round((((bearingDeg % 360) + 360) % 360) / 45) % 8;
  return COMPASS[i];
}

export function describeDistance(hit: Pick<NearestOceanResponse, "distance_km" | "bearing_deg">): string {
  return `${Math.round(hit.distance_km)} km ${compassWord(hit.bearing_deg)} of you`;
}

type Status =
  | { kind: "idle" }
  | { kind: "locating" }
  | { kind: "searching" }
  | { kind: "denied" }
  | { kind: "outside" }
  | { kind: "error"; message: string }
  | { kind: "found"; hit: NearestOceanResponse; withBearing: boolean };

type OrientationCtor = typeof DeviceOrientationEvent & { requestPermission?: () => Promise<"granted" | "denied"> };

/** Read one compass heading (0 = north), or null after a short wait / no sensor. */
function readHeading(timeoutMs = 1200): Promise<number | null> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || typeof DeviceOrientationEvent === "undefined") return resolve(null);
    let done = false;
    const finish = (v: number | null) => {
      if (done) return;
      done = true;
      window.removeEventListener("deviceorientationabsolute", onEvent as EventListener);
      window.removeEventListener("deviceorientation", onEvent as EventListener);
      resolve(v);
    };
    const onEvent = (e: DeviceOrientationEvent & { webkitCompassHeading?: number }) => {
      if (typeof e.webkitCompassHeading === "number") return finish(e.webkitCompassHeading);
      if (e.absolute && typeof e.alpha === "number") return finish((360 - e.alpha) % 360);
    };
    window.addEventListener("deviceorientationabsolute", onEvent as EventListener);
    window.addEventListener("deviceorientation", onEvent as EventListener);
    setTimeout(() => finish(null), timeoutMs);
  });
}

function getPosition(): Promise<GeolocationPosition> {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) return reject(new Error("no_geolocation"));
    navigator.geolocation.getCurrentPosition(resolve, reject, { enableHighAccuracy: false, timeout: 15_000, maximumAge: 60_000 });
  });
}

export function BeachMode({ meta }: { meta: MetaResponse }) {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [guideOpen, setGuideOpen] = useState(false);

  const point = useCallback(async () => {
    // iOS needs the motion permission from inside the tap itself, before any await
    const Ctor = (typeof DeviceOrientationEvent !== "undefined" ? DeviceOrientationEvent : undefined) as
      | OrientationCtor
      | undefined;
    const motionPermission = Ctor?.requestPermission?.().catch(() => "denied" as const);

    setStatus({ kind: "locating" });
    let pos: GeolocationPosition;
    try {
      pos = await getPosition();
    } catch {
      setStatus({ kind: "denied" });
      return;
    }
    const motionOk = motionPermission ? (await motionPermission) === "granted" : true;
    const heading = motionOk ? await readHeading() : null;

    setStatus({ kind: "searching" });
    const lat = pos.coords.latitude;
    const lon = pos.coords.longitude;
    try {
      const hit = await apiClient.getNearestOcean({ lat, lon, bearing_deg: heading });
      setStatus({ kind: "found", hit, withBearing: heading !== null });
    } catch (err) {
      if (err instanceof ApiError && err.code === "out_of_region") return setStatus({ kind: "outside" });
      if (err instanceof ApiError && err.code === "no_ocean_in_view" && heading !== null) {
        // Nothing that way: fall back to the nearest ocean cell in any direction
        try {
          const hit = await apiClient.getNearestOcean({ lat, lon });
          return setStatus({ kind: "found", hit, withBearing: false });
        } catch {
          /* fall through */
        }
      }
      setStatus({
        kind: "error",
        message:
          err instanceof ApiError && err.status === 503
            ? "Beach mode is switched off at the moment."
            : "Couldn't find the nearest ocean right now. Please try again.",
      });
    }
  }, []);

  const busy = status.kind === "locating" || status.kind === "searching";
  const hit = status.kind === "found" ? status.hit : null;

  return (
    <div className="flex-1 flex flex-col items-center px-4 py-6 gap-5 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <h1 className="text-xl font-bold">Beach mode</h1>
      <p className="text-sm text-center text-slate-600 dark:text-slate-400 max-w-sm">
        Stand on the shore, face the water and tap. We use your location once to find the sea in front of you.
        Nothing is saved.
      </p>

      <button
        type="button"
        onClick={() => void point()}
        disabled={busy}
        className="w-full max-w-sm rounded-2xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-60 text-white text-lg font-semibold py-6 inline-flex items-center justify-center gap-2 shadow-xl"
      >
        {busy ? <Loader2 className="h-6 w-6 animate-spin" /> : <Compass className="h-6 w-6" />}
        {status.kind === "locating" ? "Finding you…" : status.kind === "searching" ? "Looking out to sea…" : "Point at the sea"}
      </button>

      {status.kind === "denied" && (
        <p className="text-sm text-center max-w-sm">
          Location is off, so we can&apos;t point for you.{" "}
          <Link href="/explore" className="underline text-cyan-700 dark:text-cyan-400">
            Pick a spot on the map instead
          </Link>
          .
        </p>
      )}
      {status.kind === "outside" && (
        <p className="text-sm text-center max-w-sm">
          OceanScope covers the North Indian Ocean.{" "}
          <Link href="/explore" className="underline text-cyan-700 dark:text-cyan-400">
            Open the map to see the area we cover
          </Link>
          .
        </p>
      )}
      {status.kind === "error" && <p className="text-sm text-center max-w-sm">{status.message}</p>}

      {hit && status.kind === "found" && (
        <BeachResult meta={meta} hit={hit} withBearing={status.withBearing} onAsk={meta.features.chat ? () => setGuideOpen(true) : undefined} />
      )}

      {hit && guideOpen && meta.features.chat && (
        <GuideDrawer
          meta={meta}
          state={{
            date: hit.date,
            depth: findClosestDepth(0.5, meta.depths_m),
            field: "pred",
            lat: hit.lat,
            lon: hit.lon,
            view: "map",
          }}
          onNavigate={() => ({ summary: null, undoToken: null })}
          onUndo={() => {}}
          onClose={() => setGuideOpen(false)}
        />
      )}
    </div>
  );
}

function BeachResult({
  meta,
  hit,
  withBearing,
  onAsk,
}: {
  meta: MetaResponse;
  hit: NearestOceanResponse;
  withBearing: boolean;
  onAsk?: () => void;
}) {
  const { data: profile, isLoading, isError } = useProfile({ lat: hit.lat, lon: hit.lon, date: hit.date });
  const values = (profile?.pred ?? []).filter((v): v is number => v !== null);
  const vmin = values.length ? Math.min(...values) : 0;
  const vmax = values.length ? Math.max(...values) : 1;

  return (
    <section aria-label="The sea in front of you" className="w-full max-w-sm space-y-3">
      <div className="text-center">
        <p className="text-lg font-semibold">
          {withBearing ? describeDistance(hit) : `Nearest sea: ${describeDistance(hit)}`}
        </p>
        <p className="text-xs text-slate-500 inline-flex items-center gap-1">
          <MapPin className="h-3 w-3" /> {formatCoord(hit.lat, hit.lon)} · {formatDate(hit.date)}
        </p>
      </div>

      {isLoading && <p className="text-sm text-center">Loading the water column…</p>}
      {isError && <p className="text-sm text-center">Couldn&apos;t load the temperature profile.</p>}
      {profile && (
        <ol aria-label="Temperature from the surface down" className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800">
          {profile.depths_m.map((d, i) => {
            const v = profile.pred[i];
            return (
              <li
                key={d}
                className="flex items-center justify-between px-3 py-2 text-sm font-medium"
                style={{ background: getColorHex(v, vmin, vmax), color: v !== null && (v - vmin) / (vmax - vmin || 1) > 0.55 ? "#0f172a" : "#f8fafc" }}
              >
                <span>{formatDepth(d)}</span>
                <span>{formatTemp(v)}</span>
              </li>
            );
          })}
        </ol>
      )}

      <div className="flex gap-2">
        {onAsk && (
          <button
            type="button"
            onClick={onAsk}
            className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-lg bg-slate-900 dark:bg-white text-white dark:text-slate-900 text-sm font-medium py-2.5"
          >
            <MessageCircle className="h-4 w-4" /> Ask about it
          </button>
        )}
        <Link
          href={`/explore?date=${hit.date}&lat=${hit.lat}&lon=${hit.lon}`}
          className="flex-1 text-center rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-medium py-2.5"
        >
          Open in the map
        </Link>
      </div>
      {meta.synthetic && <p className="text-[11px] text-center text-slate-500">{meta.caveats.synthetic}</p>}
    </section>
  );
}

export default BeachMode;
