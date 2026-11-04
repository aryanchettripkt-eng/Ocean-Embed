"use client";

/**
 * Ocean postcards (PRD_FRONTEND §3.4d). Sends only date, depth and an optional curated
 * event id to /postcard. The picture never carries numbers: the caption is HTML from the
 * response, with POSTCARD_CAVEAT under it. "Download" composes image + caption + caveat
 * on a canvas so they stay attached; "Copy link" shares the explorer URL, not the image.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Check, Download, Link2, X } from "lucide-react";

import { ApiError, apiClient, resolveApiUrl, type PostcardResponse } from "../../lib/api/client";
import { formatDate, formatDepth, formatTemp } from "../../lib/format";
import { TypingDots } from "../common/TypingDots";

export const POSTCARD_TIMEOUT_MS = 45_000;

export function postcardErrorMessage(err: unknown, timedOut: boolean): string {
  if (timedOut) return "The postcard took too long to print. Please try again.";
  if (err instanceof ApiError) {
    if (err.status === 429) return "Lots of postcards are printing right now. Try again in a minute.";
    if (err.status === 503) return "Postcards are switched off at the moment.";
    if (err.status === 502) return "The image service didn't answer. Please try again shortly.";
    if (err.status === 404) return "This date or event isn't available for a postcard.";
  }
  return "Couldn't make a postcard. Please try again.";
}

export function captionLines(r: PostcardResponse): string[] {
  const c = r.caption;
  const lines = [`${formatDate(c.date)} · ${formatDepth(c.depth_m)}`];
  if (c.mean_c !== null && c.mean_c !== undefined) lines.push(`Mean temperature ${formatTemp(c.mean_c)}`);
  lines.push(c.source);
  return lines;
}

interface PostcardPanelProps {
  date: string;
  depth: number;
  eventId?: string | null;
  eventTitle?: string | null;
  caveat: string;
  onClose: () => void;
}

export function PostcardPanel({ date, depth, eventId, eventTitle, caveat, onClose }: PostcardPanelProps) {
  const [result, setResult] = useState<PostcardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [copied, setCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const make = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, POSTCARD_TIMEOUT_MS);
    setPending(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiClient.postPostcard(
        { date, depth_m: depth, ...(eventId ? { event_id: eventId } : {}) },
        controller.signal
      );
      if (abortRef.current === controller) setResult(res);
    } catch (err) {
      // A superseded request (e.g. aborted by an effect re-run) must not overwrite the current one
      if (abortRef.current !== controller) return;
      if (!controller.signal.aborted || timedOut) setError(postcardErrorMessage(err, timedOut));
      else setError("Cancelled.");
    } finally {
      clearTimeout(timer);
      if (abortRef.current === controller) {
        abortRef.current = null;
        setPending(false);
      }
    }
  }, [date, depth, eventId]);

  useEffect(() => {
    void make();
    return () => abortRef.current?.abort();
  }, [make]);

  const download = useCallback(async () => {
    if (!result) return;
    const blob = await fetch(resolveApiUrl(result.image_url)).then((r) => r.blob());
    const bmp = await createImageBitmap(blob);
    const pad = 32;
    const w = bmp.width + pad * 2;
    const lines = [...captionLines(result), result.label];
    const textH = 28 * lines.length + 22 * Math.ceil(caveat.length / 90) + pad;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = bmp.height + pad * 2 + textH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#f5efe2";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bmp, pad, pad);
    let y = bmp.height + pad * 2 + 8;
    ctx.fillStyle = "#1e293b";
    ctx.font = "600 20px system-ui, sans-serif";
    lines.forEach((l, i) => {
      if (i === 1) ctx.font = "18px system-ui, sans-serif";
      ctx.fillText(l, pad, y);
      y += 28;
    });
    ctx.font = "14px system-ui, sans-serif";
    ctx.fillStyle = "#475569";
    const words = caveat.split(" ");
    let line = "";
    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > w - pad * 2) {
        ctx.fillText(line, pad, y);
        y += 22;
        line = word;
      } else line = test;
    }
    if (line) ctx.fillText(line, pad, y);
    canvas.toBlob((png) => {
      if (!png) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(png);
      a.download = `oceanscope-postcard-${result.caption.date}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  }, [result, caveat]);

  return (
    <section
      aria-label="Ocean postcard"
      className="hud-glass flex max-h-[calc(100dvh-12rem)] w-full flex-col overflow-hidden rounded-[20px] sm:w-[380px]"
    >
      <header className="hud-divider flex shrink-0 items-center justify-between gap-3 border-b px-4 py-3">
        <h2 className="truncate text-sm font-semibold text-neutral-950 dark:text-white">
          Ocean postcard{eventTitle ? ` · ${eventTitle}` : ""}
        </h2>
        <button type="button" onClick={onClose} aria-label="Close postcard" className="hud-icon-btn">
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 space-y-2.5 overflow-y-auto overscroll-contain p-4 text-neutral-700 dark:text-neutral-300">
        {pending && (
          <div role="status" className="space-y-2">
            <div className="relative h-40 overflow-hidden rounded-xl bg-[#f5efe2] p-2 dark:bg-[#24201a]">
              <div className="h-full rounded-lg bg-white/70 motion-safe:animate-[pulse_1.6s_ease-in-out_infinite] dark:bg-amber-100/10" />
            </div>
            <div className="flex items-center gap-2.5 text-xs">
              <TypingDots label="Printing" className="text-neutral-500" />
              <span>Printing your postcard… this can take up to 45 seconds.</span>
              <button type="button" onClick={() => abortRef.current?.abort()} className="ml-auto text-neutral-500 underline underline-offset-2 hover:text-neutral-900 dark:hover:text-white">
                Cancel
              </button>
            </div>
          </div>
        )}

        {error && !pending && (
          <div className="text-xs space-y-2">
            <p>{error}</p>
            <button type="button" onClick={() => void make()} className="hud-btn h-7 border border-black/10 dark:border-white/10">
              Try again
            </button>
          </div>
        )}

        {result && (
          <figure className="space-y-2.5">
            {/* eslint-disable-next-line @next/next/no-img-element -- backend-served image, not a static asset */}
            <img
              src={resolveApiUrl(result.image_url)}
              alt={`Artistic postcard of the ocean on ${formatDate(result.caption.date)}`}
              className="w-full rounded-xl bg-[#f5efe2] p-2 dark:bg-[#24201a]"
            />
            <figcaption className="space-y-0.5 text-xs">
              {captionLines(result).map((l, i) => (
                <p key={i} className={i === 0 ? "text-sm font-semibold text-neutral-950 dark:text-white" : ""}>
                  {l}
                </p>
              ))}
              <p className="text-[11px] text-neutral-500 dark:text-neutral-400">{result.label}</p>
            </figcaption>
            <p className="text-[10px] leading-snug text-neutral-500">{caveat}</p>
            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={() => void download()}
                className="hud-btn hud-btn-active"
              >
                <Download aria-hidden="true" className="h-3.5 w-3.5" /> Download
              </button>
              <button
                type="button"
                onClick={() => {
                  void navigator.clipboard?.writeText(window.location.href).then(() => setCopied(true));
                }}
                className="hud-btn border border-black/10 dark:border-white/10"
              >
                {copied ? <Check aria-hidden="true" className="h-3.5 w-3.5" /> : <Link2 aria-hidden="true" className="h-3.5 w-3.5" />}
                {copied ? "Link copied" : "Copy link"}
              </button>
            </div>
          </figure>
        )}
      </div>
    </section>
  );
}

export default PostcardPanel;
