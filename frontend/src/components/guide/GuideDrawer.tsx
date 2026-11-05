"use client";

/**
 * Ocean guide = chat + navigator (PRD_FRONTEND §3.4a). Loaded with `import()` on first open,
 * so none of this is in the initial bundle. Shown only when `meta.features.chat` is true.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, ChevronDown, ChevronRight, ShieldQuestion, Undo2, X } from "lucide-react";

import {
  ApiError,
  apiClient,
  type ChatMessage,
  type ChatResponse,
  type MetaResponse,
} from "../../lib/api/client";
import type { ExplorerState } from "../../lib/url-state";
import { formatDepth } from "../../lib/format";
import { MicButton } from "./MicButton";
import { TypingDots } from "../common/TypingDots";

export const CHAT_TIMEOUT_MS = 25_000;
const MAX_HISTORY = 10;
const MAX_CHARS = 2000;

export interface NavigateResult {
  /** "Moved to …" chip text, or null when nothing moved. */
  summary: string | null;
  /** Token to pass to `onUndo`; null when there is nothing to undo. */
  undoToken: number | null;
}

export interface GuidePrompt {
  text: string;
  /** Changes on every request so the same text can be sent twice. */
  nonce: number;
}

interface Turn {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
  nav?: NavigateResult;
  undone?: boolean;
  error?: boolean;
}

export interface GuideDrawerProps {
  meta: MetaResponse;
  state: ExplorerState;
  /** Titles from /events for a starter chip; never typed into the frontend. */
  eventTitles?: string[];
  onNavigate: (response: ChatResponse) => NavigateResult;
  onUndo: (token: number) => void;
  onClose: () => void;
  /** A message to send as soon as it arrives (e.g. the time machine's "Explain"). */
  prompt?: GuidePrompt | null;
  /** Positioning classes; defaults to a fixed bottom sheet / right-hand card. */
  placementClassName?: string;
}

const DEFAULT_PLACEMENT =
  "fixed inset-x-3 bottom-3 h-[78vh] md:inset-x-auto md:right-4 md:top-20 md:bottom-4 md:h-auto md:w-[380px]";

export function guideErrorMessage(err: unknown): string {
  if (err instanceof DOMException && err.name === "TimeoutError") {
    return "The guide took too long to answer. Please try again.";
  }
  if (err instanceof ApiError) {
    if (err.status === 429) return "The guide is getting a lot of questions right now. Try again in a minute.";
    if (err.status === 503) return "The guide is switched off at the moment. The maps and charts still work.";
    if (err.status === 502) return "The guide couldn't reach its AI service. Please try again shortly.";
    if (err.status === 422) return "That message couldn't be sent. Try a shorter question.";
  }
  return "Something went wrong while asking the guide. Please try again.";
}

export function starterQuestions(state: ExplorerState, eventTitles: string[] = []): string[] {
  const out = [
    state.lat !== null
      ? `What's happening at ${formatDepth(state.depth)} here?`
      : `What's happening at ${formatDepth(state.depth)} in the Bay of Bengal?`,
    "Show me the warmest water today",
  ];
  if (eventTitles.length > 0) out.push(`Take me to ${eventTitles[0]}`);
  return out;
}

export function GuideDrawer({
  meta,
  state,
  eventTitles = [],
  onNavigate,
  onUndo,
  onClose,
  prompt,
  placementClassName = DEFAULT_PLACEMENT,
}: GuideDrawerProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const abortRef = useRef<{ controller: AbortController; timedOut: boolean } | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const lastPromptNonce = useRef<number | null>(null);

  // Keep latest values for the async send without re-creating it on every map move
  const stateRef = useRef(state);
  stateRef.current = state;
  const turnsRef = useRef(turns);
  turnsRef.current = turns;
  const navigateRef = useRef(onNavigate);
  navigateRef.current = onNavigate;

  useEffect(() => {
    listRef.current?.scrollTo?.({ top: listRef.current.scrollHeight });
  }, [turns, pending]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Cancel an in-flight request when the drawer unmounts
  useEffect(() => () => abortRef.current?.controller.abort(), []);

  const send = useCallback(async (raw: string) => {
    const text = raw.trim().slice(0, MAX_CHARS);
    if (!text || abortRef.current) return;

    const history: ChatMessage[] = [
      ...turnsRef.current
        .filter((t) => !t.error)
        .map((t) => ({ role: t.role, content: t.content.slice(0, MAX_CHARS) })),
      { role: "user" as const, content: text },
    ].slice(-MAX_HISTORY);
    // The backend expects the conversation to start with a user turn
    while (history.length > 1 && history[0].role !== "user") history.shift();

    setTurns((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setPending(true);

    const controller = new AbortController();
    const handle = { controller, timedOut: false };
    abortRef.current = handle;
    const timer = setTimeout(() => {
      handle.timedOut = true;
      controller.abort();
    }, CHAT_TIMEOUT_MS);

    const s = stateRef.current;
    try {
      const response = await apiClient.postChat(
        {
          messages: history,
          context: { date: s.date, depth_m: s.depth, lat: s.lat, lon: s.lon },
        },
        controller.signal
      );
      const nav = response.grounded ? navigateRef.current(response) : undefined;
      setTurns((prev) => [...prev, { role: "assistant", content: response.answer, response, nav }]);
    } catch (err) {
      if (controller.signal.aborted && !handle.timedOut) {
        setTurns((prev) => [...prev, { role: "assistant", content: "Cancelled.", error: true }]);
      } else {
        const e = handle.timedOut ? new DOMException("timeout", "TimeoutError") : err;
        setTurns((prev) => [...prev, { role: "assistant", content: guideErrorMessage(e), error: true }]);
      }
    } finally {
      clearTimeout(timer);
      abortRef.current = null;
      setPending(false);
    }
  }, []);

  useEffect(() => {
    if (prompt && prompt.nonce !== lastPromptNonce.current) {
      lastPromptNonce.current = prompt.nonce;
      void send(prompt.text);
    }
  }, [prompt, send]);

  const providerLabel = meta.features.provider_label ?? meta.chat.provider_label ?? null;
  const starters = starterQuestions(state, eventTitles);

  return (
    <aside
      role="dialog"
      aria-label="Ask the ocean"
      className={`hud-glass z-50 flex flex-col overflow-hidden rounded-[20px] ${placementClassName}`}
    >
      <header className="hud-divider flex shrink-0 items-start justify-between gap-3 border-b px-4 pb-3 pt-3.5">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-neutral-950 dark:text-white">Ask the ocean</h2>
          <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
            Ask a question and the map moves to show you.
          </p>
        </div>
        <button type="button" onClick={onClose} aria-label="Close guide" className="hud-icon-btn">
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </header>

      <div
        ref={listRef}
        className="min-h-0 flex-1 space-y-3 overflow-y-auto overscroll-contain px-4 py-3.5"
        aria-live="polite"
      >
        {turns.length === 0 && (
          <div className="space-y-2">
            <p className="hud-label">Try one of these</p>
            {starters.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => void send(q)}
                disabled={pending}
                className="block w-full rounded-2xl bg-black/[0.035] px-3.5 py-2.5 text-left text-xs text-neutral-800 ring-1 ring-inset ring-black/[0.05] transition-colors hover:bg-black/[0.06] disabled:opacity-50 dark:bg-white/[0.04] dark:text-neutral-200 dark:ring-white/[0.07] dark:hover:bg-white/[0.08]"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {turns.map((t, i) => (
          <TurnView
            key={i}
            turn={t}
            onUndo={() => {
              if (t.nav?.undoToken != null) {
                onUndo(t.nav.undoToken);
                setTurns((prev) => prev.map((p, j) => (j === i ? { ...p, undone: true } : p)));
              }
            }}
          />
        ))}

        {pending && (
          <div className="mr-10 flex items-center gap-3" role="status">
            <div className="rounded-2xl rounded-bl-md bg-black/[0.04] px-3.5 py-3 text-neutral-500 dark:bg-white/[0.06] dark:text-neutral-400">
              <TypingDots label="Looking at the data…" />
            </div>
            <button
              type="button"
              onClick={() => abortRef.current?.controller.abort()}
              className="text-[11px] text-neutral-500 underline underline-offset-2 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white"
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      <form
        className="hud-divider shrink-0 space-y-2 border-t p-3"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="flex items-end gap-1.5 rounded-2xl bg-black/[0.04] p-1.5 ring-1 ring-inset ring-black/[0.05] focus-within:ring-cyan-500/60 dark:bg-white/[0.05] dark:ring-white/[0.08]">
          <label htmlFor="guide-input" className="sr-only">
            Your question
          </label>
          <textarea
            id="guide-input"
            ref={inputRef}
            rows={2}
            maxLength={MAX_CHARS}
            value={input}
            disabled={pending}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            placeholder="Ask about the ocean here…"
            className="min-w-0 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none disabled:opacity-60 dark:text-neutral-100 dark:placeholder:text-neutral-500"
          />
          {meta.features.voice && (
            <MicButton disabled={pending} onTranscript={setInput} onAutoSend={(text) => void send(text)} />
          )}
          <button
            type="submit"
            disabled={pending || !input.trim()}
            aria-label="Send"
            className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-cyan-400 text-neutral-950 transition-colors hover:bg-cyan-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-cyan-400 disabled:bg-black/10 disabled:text-neutral-400 dark:disabled:bg-white/10 dark:disabled:text-neutral-500"
          >
            <ArrowUp aria-hidden="true" className="h-4 w-4" />
          </button>
        </div>
        {providerLabel && <p className="px-1 text-[10px] text-neutral-500 dark:text-neutral-400">{providerLabel}</p>}
        <p className="px-1 text-[10px] leading-snug text-neutral-500 dark:text-neutral-400">{meta.caveats.ai}</p>
      </form>
    </aside>
  );
}

function TurnView({ turn, onUndo }: { turn: Turn; onUndo: () => void }) {
  const [showData, setShowData] = useState(false);

  if (turn.role === "user") {
    return (
      <div className="ml-10 flex justify-end">
        <div className="whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-neutral-900 px-3.5 py-2 text-sm text-white dark:bg-white dark:text-neutral-950">
          {turn.content}
        </div>
      </div>
    );
  }

  const r = turn.response;
  const ungrounded = r && r.grounded === false;
  return (
    <div className="mr-6 space-y-1.5">
      <div
        className={`whitespace-pre-wrap break-words rounded-2xl rounded-bl-md px-3.5 py-2 text-sm ${
          turn.error
            ? "bg-amber-500/10 text-amber-900 ring-1 ring-inset ring-amber-500/25 dark:text-amber-100"
            : "bg-black/[0.04] text-neutral-900 dark:bg-white/[0.06] dark:text-neutral-100"
        }`}
      >
        {ungrounded && (
          <span className="mb-1 inline-flex items-center gap-1 rounded-full bg-black/[0.06] px-2 py-0.5 text-[10px] font-medium text-neutral-700 dark:bg-white/[0.08] dark:text-neutral-300">
            <ShieldQuestion aria-hidden="true" className="h-3 w-3" /> Couldn&apos;t verify
          </span>
        )}
        {/* Plain text only: no HTML is ever rendered from the model (PRD_FRONTEND §4 Security) */}
        <div>{turn.content}</div>
      </div>

      {turn.nav?.summary && (
        <div className="flex items-center gap-2 text-[11px] text-neutral-600 dark:text-neutral-300">
          <span className="rounded-full bg-cyan-500/10 px-2.5 py-0.5 ring-1 ring-inset ring-cyan-500/25">
            {turn.undone ? "Moved back" : turn.nav.summary}
          </span>
          {!turn.undone && turn.nav.undoToken != null && (
            <button
              type="button"
              onClick={onUndo}
              className="inline-flex items-center gap-1 text-neutral-600 underline underline-offset-2 hover:text-neutral-950 dark:text-neutral-400 dark:hover:text-white"
            >
              <Undo2 aria-hidden="true" className="h-3 w-3" /> Undo
            </button>
          )}
        </div>
      )}

      {r && r.tool_calls.length > 0 && (
        <div className="text-[11px]">
          <button
            type="button"
            aria-expanded={showData}
            onClick={() => setShowData(!showData)}
            className="inline-flex items-center gap-1 text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white"
          >
            {showData ? <ChevronDown aria-hidden="true" className="h-3 w-3" /> : <ChevronRight aria-hidden="true" className="h-3 w-3" />}
            Data used ({r.tool_calls.length})
          </button>
          {showData && (
            <ul className="mt-1 space-y-1">
              {r.tool_calls.map((c, i) => (
                <li key={i} className="rounded-xl bg-black/[0.03] p-2 ring-1 ring-inset ring-black/[0.05] dark:bg-white/[0.03] dark:ring-white/[0.07]">
                  <p className="font-mono font-semibold text-neutral-700 dark:text-neutral-300">{c.name}</p>
                  <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all text-[10px] text-neutral-600 dark:text-neutral-400">
                    {JSON.stringify({ arguments: c.arguments, result: c.result }, null, 1)}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export default GuideDrawer;
