"use client";

/**
 * Ocean guide = chat + navigator (PRD_FRONTEND §3.4a). Loaded with `import()` on first open,
 * so none of this is in the initial bundle. Shown only when `meta.features.chat` is true.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Loader2, Send, ShieldQuestion, Undo2, X } from "lucide-react";

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
}

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
      className="fixed z-50 inset-x-0 bottom-0 h-[78vh] rounded-t-2xl md:rounded-none md:inset-x-auto md:right-0 md:top-14 md:bottom-0 md:h-auto md:w-[400px] flex flex-col bg-white dark:bg-slate-950 border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-800 shadow-2xl"
    >
      <header className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Ask the ocean</h2>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            Ask a question and the map moves to show you.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close guide"
          className="p-1.5 rounded-md text-slate-500 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-900"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div ref={listRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3" aria-live="polite">
        {turns.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs text-slate-500 dark:text-slate-400">Try one of these:</p>
            {starters.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => void send(q)}
                disabled={pending}
                className="block w-full text-left text-xs px-3 py-2 rounded-lg border border-cyan-200 dark:border-cyan-900 bg-cyan-50 dark:bg-cyan-950/40 text-cyan-900 dark:text-cyan-100 hover:bg-cyan-100 dark:hover:bg-cyan-900/50 disabled:opacity-50"
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
          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400" role="status">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>Looking at the data…</span>
            <button
              type="button"
              onClick={() => abortRef.current?.controller.abort()}
              className="ml-auto underline text-cyan-600 dark:text-cyan-400"
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      <form
        className="border-t border-slate-200 dark:border-slate-800 p-3 space-y-1.5"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="flex items-end gap-2">
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
            className="flex-1 resize-none text-sm rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 px-3 py-2 disabled:opacity-60"
          />
          {meta.features.voice && (
            <MicButton disabled={pending} onTranscript={setInput} onAutoSend={(text) => void send(text)} />
          )}
          <button
            type="submit"
            disabled={pending || !input.trim()}
            aria-label="Send"
            className="p-2.5 rounded-lg bg-cyan-600 text-white disabled:opacity-40 hover:bg-cyan-500"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        {providerLabel && (
          <p className="text-[10px] text-slate-500 dark:text-slate-400">{providerLabel}</p>
        )}
        <p className="text-[10px] leading-snug text-slate-500 dark:text-slate-400">{meta.caveats.ai}</p>
      </form>
    </aside>
  );
}

function TurnView({ turn, onUndo }: { turn: Turn; onUndo: () => void }) {
  const [showData, setShowData] = useState(false);

  if (turn.role === "user") {
    return (
      <div className="ml-8 rounded-xl rounded-br-sm bg-cyan-600 text-white text-sm px-3 py-2 whitespace-pre-wrap break-words">
        {turn.content}
      </div>
    );
  }

  const r = turn.response;
  const ungrounded = r && r.grounded === false;
  return (
    <div className="mr-4 space-y-1.5">
      <div
        className={`rounded-xl rounded-bl-sm text-sm px-3 py-2 whitespace-pre-wrap break-words ${
          turn.error
            ? "bg-amber-50 dark:bg-amber-950/40 text-amber-900 dark:text-amber-100 border border-amber-200 dark:border-amber-900"
            : "bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100"
        }`}
      >
        {ungrounded && (
          <span className="mb-1 inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
            <ShieldQuestion className="h-3 w-3" /> Couldn&apos;t verify
          </span>
        )}
        {/* Plain text only: no HTML is ever rendered from the model (PRD_FRONTEND §4 Security) */}
        <div>{turn.content}</div>
      </div>

      {turn.nav?.summary && (
        <div className="flex items-center gap-2 text-[11px] text-slate-600 dark:text-slate-300">
          <span className="px-2 py-0.5 rounded-full bg-cyan-50 dark:bg-cyan-950/50 border border-cyan-200 dark:border-cyan-900">
            {turn.undone ? "Moved back" : turn.nav.summary}
          </span>
          {!turn.undone && turn.nav.undoToken != null && (
            <button type="button" onClick={onUndo} className="inline-flex items-center gap-1 underline text-cyan-700 dark:text-cyan-400">
              <Undo2 className="h-3 w-3" /> Undo
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
            className="inline-flex items-center gap-1 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
          >
            {showData ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Data used ({r.tool_calls.length})
          </button>
          {showData && (
            <ul className="mt-1 space-y-1">
              {r.tool_calls.map((c, i) => (
                <li key={i} className="rounded-md bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-2">
                  <p className="font-mono font-semibold text-slate-700 dark:text-slate-300">{c.name}</p>
                  <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all text-[10px] text-slate-600 dark:text-slate-400">
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
