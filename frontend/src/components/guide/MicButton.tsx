"use client";

/**
 * Voice dive mic button (PRD_FRONTEND §3.4b). Hold to talk, or tap to start and tap again
 * to stop. Max 30 s with a countdown, then /transcribe; the transcript sits in the input
 * for 1.5 s (any key cancels auto-send) and is then sent to the guide as text.
 * Audio goes only to our backend and is never stored in the browser.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Mic, MicOff, Square } from "lucide-react";
import { apiClient, ApiError } from "../../lib/api/client";

export const MAX_RECORD_S = 30;
export const AUTO_SEND_MS = 1500;
const HOLD_THRESHOLD_MS = 300;
const MAX_BYTES = 1_000_000;

type Phase = "idle" | "explain" | "recording" | "transcribing" | "denied" | "countdown";

export function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/aac"];
  return options.find((t) => MediaRecorder.isTypeSupported?.(t));
}

interface MicButtonProps {
  disabled?: boolean;
  onTranscript: (text: string) => void;
  onAutoSend: (text: string) => void;
}

export function MicButton({ disabled, onTranscript, onAutoSend }: MicButtonProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [secondsLeft, setSecondsLeft] = useState(MAX_RECORD_S);
  const [message, setMessage] = useState<string | null>(null);
  const permissionAsked = useRef(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pressStartRef = useRef<number>(0);
  const autoSendRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const supported =
    typeof window !== "undefined" &&
    typeof MediaRecorder !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia;

  const cleanupStream = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  useEffect(
    () => () => {
      cleanupStream();
      abortRef.current?.abort();
      if (autoSendRef.current) clearTimeout(autoSendRef.current);
    },
    []
  );

  const transcribe = useCallback(
    async (blob: Blob) => {
      if (blob.size === 0) {
        setPhase("idle");
        return;
      }
      if (blob.size > MAX_BYTES) {
        setMessage("That clip was too long. Try a shorter question.");
        setPhase("idle");
        return;
      }
      setPhase("transcribing");
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const res = await apiClient.postTranscribe(blob, navigator.language || "en", controller.signal);
        const text = res.text.trim();
        if (!text) {
          setMessage("I didn't catch that. Try again or type your question.");
          setPhase("idle");
          return;
        }
        onTranscript(text);
        setPhase("countdown");
        const cancel = () => {
          if (autoSendRef.current) clearTimeout(autoSendRef.current);
          autoSendRef.current = null;
          setPhase("idle");
          window.removeEventListener("keydown", cancel);
        };
        window.addEventListener("keydown", cancel, { once: true });
        autoSendRef.current = setTimeout(() => {
          window.removeEventListener("keydown", cancel);
          autoSendRef.current = null;
          setPhase("idle");
          onAutoSend(text);
        }, AUTO_SEND_MS);
      } catch (err) {
        if (!controller.signal.aborted) {
          setMessage(
            err instanceof ApiError && err.status === 429
              ? "Voice is busy right now. Please type your question."
              : "Couldn't turn that into text. Please type your question."
          );
        }
        setPhase("idle");
      } finally {
        abortRef.current = null;
      }
    },
    [onTranscript, onAutoSend]
  );

  const stop = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, []);

  const start = useCallback(async () => {
    setMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = pickMimeType();
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        cleanupStream();
        recorderRef.current = null;
        void transcribe(new Blob(chunksRef.current, { type: rec.mimeType || mimeType || "audio/webm" }));
      };
      recorderRef.current = rec;
      rec.start();
      setSecondsLeft(MAX_RECORD_S);
      setPhase("recording");
      const startedAt = Date.now();
      timerRef.current = setInterval(() => {
        const left = MAX_RECORD_S - Math.floor((Date.now() - startedAt) / 1000);
        setSecondsLeft(Math.max(0, left));
        if (left <= 0) stop();
      }, 250);
    } catch {
      cleanupStream();
      setPhase("denied");
    }
  }, [stop, transcribe]);

  if (!supported) return null;

  const onPointerDown = () => {
    if (disabled || phase === "transcribing") return;
    if (!permissionAsked.current) {
      setPhase("explain");
      return;
    }
    if (phase === "recording") return; // the tap that stops is handled on pointer up
    pressStartRef.current = Date.now();
    void start();
  };

  const onPointerUp = () => {
    if (phase !== "recording") return;
    const held = Date.now() - pressStartRef.current;
    // Long press = hold-to-talk (stop on release); short tap = toggle (stop on the next tap)
    if (held >= HOLD_THRESHOLD_MS || pressStartRef.current === 0) stop();
    pressStartRef.current = 0;
  };

  const label =
    phase === "recording"
      ? `Stop recording (${secondsLeft} s left)`
      : phase === "denied"
        ? "Microphone blocked"
        : "Hold to talk, or tap to start";

  return (
    <div className="relative">
      <button
        type="button"
        aria-label={label}
        title={label}
        disabled={disabled || phase === "transcribing"}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onKeyDown={(e) => {
          if (e.key !== "Enter" && e.key !== " ") return;
          e.preventDefault();
          if (phase === "recording") stop();
          else if (!permissionAsked.current) setPhase("explain");
          else void start();
        }}
        className={`p-2.5 rounded-lg border touch-none select-none ${
          phase === "recording"
            ? "bg-red-600 border-red-500 text-white animate-pulse"
            : "bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200"
        } disabled:opacity-40`}
      >
        {phase === "transcribing" ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : phase === "recording" ? (
          <Square className="h-4 w-4" />
        ) : phase === "denied" ? (
          <MicOff className="h-4 w-4" />
        ) : (
          <Mic className="h-4 w-4" />
        )}
      </button>

      {(phase === "explain" || phase === "denied" || phase === "recording" || phase === "countdown" || message) && (
        <div className="absolute bottom-full right-0 mb-2 w-60 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-2.5 text-[11px] text-slate-700 dark:text-slate-200 shadow-xl">
          {phase === "explain" && (
            <>
              <p>Ask out loud: your browser will ask to use the microphone. Audio goes only to our server to turn it into text.</p>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  className="px-2 py-1 rounded bg-cyan-600 text-white"
                  onClick={() => {
                    permissionAsked.current = true;
                    pressStartRef.current = 0;
                    void start();
                  }}
                >
                  Start talking
                </button>
                <button type="button" className="px-2 py-1 underline" onClick={() => setPhase("idle")}>
                  Not now
                </button>
              </div>
            </>
          )}
          {phase === "denied" && (
            <p>
              The microphone is blocked. Allow it from the lock icon next to the address bar, then try again. You can
              still type your question.
            </p>
          )}
          {phase === "recording" && <p>Listening… {secondsLeft} s left. Release or tap again to stop.</p>}
          {phase === "countdown" && <p>Sending in a moment. Press any key to edit first.</p>}
          {phase === "idle" && message && <p>{message}</p>}
        </div>
      )}
    </div>
  );
}
