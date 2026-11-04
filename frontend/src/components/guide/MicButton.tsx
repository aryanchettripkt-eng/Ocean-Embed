"use client";

/**
 * Voice dive mic button (PRD_FRONTEND §3.4b). Hold to talk, or tap to start and tap again
 * to stop. Max 30 s with a countdown, then /transcribe; the transcript sits in the input
 * for 1.5 s (any key cancels auto-send) and is then sent to the guide as text.
 * Audio goes only to our backend and is never stored in the browser. The clip is converted to
 * 16 kHz mono WAV in memory first, the only format Muse Voice Transcribe's file endpoint takes.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Mic, MicOff, Square } from "lucide-react";
import { apiClient, ApiError } from "../../lib/api/client";
import { recordingToWav } from "../../lib/wav";
import { TypingDots } from "../common/TypingDots";
import { useLoop } from "../common/Motion";

export const MAX_RECORD_S = 30;
export const AUTO_SEND_MS = 1500;
const HOLD_THRESHOLD_MS = 300;
const MAX_BYTES = 1_000_000;

type Phase = "idle" | "explain" | "recording" | "transcribing" | "denied" | "countdown";

export function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
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
  const buttonRef = useRef<HTMLButtonElement>(null);
  useLoop(buttonRef, "pulse", phase === "recording");

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
    async (recording: Blob) => {
      if (recording.size === 0) {
        setPhase("idle");
        return;
      }
      setPhase("transcribing");
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const blob = await recordingToWav(recording);
        if (blob.size > MAX_BYTES) {
          setMessage("That clip was too long. Try a shorter question.");
          setPhase("idle");
          return;
        }
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
        ref={buttonRef}
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
        className={`inline-flex h-9 w-9 shrink-0 touch-none select-none items-center justify-center rounded-full transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-cyan-400 disabled:opacity-40 ${
          phase === "recording"
            ? "bg-red-600 text-white"
            : "text-neutral-600 hover:bg-black/[0.06] hover:text-neutral-950 dark:text-neutral-300 dark:hover:bg-white/[0.08] dark:hover:text-white"
        }`}
      >
        {phase === "transcribing" ? (
          <TypingDots label="Transcribing" />
        ) : phase === "recording" ? (
          <Square aria-hidden="true" className="h-4 w-4" />
        ) : phase === "denied" ? (
          <MicOff aria-hidden="true" className="h-4 w-4" />
        ) : (
          <Mic aria-hidden="true" className="h-4 w-4" />
        )}
      </button>

      {(phase === "explain" || phase === "denied" || phase === "recording" || phase === "countdown" || message) && (
        <div className="hud-glass absolute bottom-full right-0 mb-2 w-60 rounded-2xl p-3 text-[11px] leading-snug">
          {phase === "explain" && (
            <>
              <p>Ask out loud: your browser will ask to use the microphone. Audio goes only to our server to turn it into text.</p>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  className="hud-btn hud-btn-active h-7"
                  onClick={() => {
                    permissionAsked.current = true;
                    pressStartRef.current = 0;
                    void start();
                  }}
                >
                  Start talking
                </button>
                <button type="button" className="hud-btn h-7" onClick={() => setPhase("idle")}>
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
