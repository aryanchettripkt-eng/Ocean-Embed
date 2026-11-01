"use client";

/**
 * Event time machine (PRD_FRONTEND §3.4c). The list comes from /events only: no event text
 * is typed into the frontend, and events in the sealed test period never appear because
 * the backend filters them. Before / During / After set the date, zoom to the event bbox
 * and pick the first suggested depth; the 3D view follows the same state.
 */

import React from "react";
import { ArrowLeft, CalendarRange, ExternalLink, Sparkles, X } from "lucide-react";

import type { EventItem, MetaResponse } from "../../lib/api/client";
import type { EventPhase } from "../../lib/guide/ui-actions";
import { phaseDate } from "../../lib/guide/ui-actions";
import { formatDate } from "../../lib/format";

const PHASES: { id: EventPhase; label: string }[] = [
  { id: "before", label: "Before" },
  { id: "during", label: "During" },
  { id: "after", label: "After" },
];

export function isDateInData(date: string, meta: MetaResponse): boolean {
  return date >= meta.available_dates.start && date <= meta.available_dates.end;
}

/** Days from start to end inclusive, as ISO dates (capped so a long event stays light). */
export function eventDays(event: EventItem, maxDays = 400): string[] {
  const out: string[] = [];
  const start = Date.parse(event.start + "T00:00:00Z");
  const end = Date.parse(event.end + "T00:00:00Z");
  for (let t = start; t <= end && out.length < maxDays; t += 86_400_000) {
    out.push(new Date(t).toISOString().slice(0, 10));
  }
  return out;
}

interface TimeMachineProps {
  meta: MetaResponse;
  events: EventItem[];
  isLoading?: boolean;
  isError?: boolean;
  active: { event: EventItem; phase: EventPhase } | null;
  currentDate: string;
  onOpen: (event: EventItem, phase: EventPhase, opts?: { pickDepth?: boolean }) => void;
  onSetDate: (date: string) => void;
  onBack: () => void;
  onClose: () => void;
  onExplain?: (event: EventItem) => void;
}

export function TimeMachine({
  meta,
  events,
  isLoading,
  isError,
  active,
  currentDate,
  onOpen,
  onSetDate,
  onBack,
  onClose,
  onExplain,
}: TimeMachineProps) {
  return (
    <section
      aria-label="Event time machine"
      className="w-full sm:w-[380px] max-h-[60vh] overflow-y-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-950/95 backdrop-blur-xl shadow-2xl"
    >
      <header className="flex items-center justify-between gap-2 px-3 py-2.5 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-2 min-w-0">
          {active && (
            <button type="button" onClick={onBack} aria-label="Back to events" className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-900">
              <ArrowLeft className="h-4 w-4 text-slate-500" />
            </button>
          )}
          <CalendarRange className="h-4 w-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
            {active ? active.event.title : "Ocean events"}
          </h2>
        </div>
        <button type="button" onClick={onClose} aria-label="Close events" className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-900">
          <X className="h-4 w-4 text-slate-500" />
        </button>
      </header>

      {!active && (
        <div className="p-3 space-y-2">
          {isLoading && <p className="text-xs text-slate-500">Loading events…</p>}
          {isError && <p className="text-xs text-red-600 dark:text-red-300">Couldn&apos;t load events. Try again later.</p>}
          {!isLoading && !isError && events.length === 0 && (
            <p className="text-xs text-slate-500">No events are available yet.</p>
          )}
          {events.map((e) => (
            <button
              key={e.id}
              type="button"
              onClick={() => onOpen(e, "before", { pickDepth: true })}
              className="block w-full text-left rounded-lg border border-slate-200 dark:border-slate-800 px-3 py-2 hover:border-cyan-400 hover:bg-cyan-50/60 dark:hover:bg-cyan-950/30"
            >
              <p className="text-sm font-medium text-slate-900 dark:text-slate-100">{e.title}</p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                {formatDate(e.start)} – {formatDate(e.end)} · {e.source_name}
              </p>
            </button>
          ))}
        </div>
      )}

      {active && (
        <EventCompare
          meta={meta}
          event={active.event}
          phase={active.phase}
          currentDate={currentDate}
          onPhase={(p) => onOpen(active.event, p, { pickDepth: false })}
          onSetDate={onSetDate}
          onExplain={onExplain}
        />
      )}
    </section>
  );
}

function EventCompare({
  meta,
  event,
  phase,
  currentDate,
  onPhase,
  onSetDate,
  onExplain,
}: {
  meta: MetaResponse;
  event: EventItem;
  phase: EventPhase;
  currentDate: string;
  onPhase: (p: EventPhase) => void;
  onSetDate: (d: string) => void;
  onExplain?: (event: EventItem) => void;
}) {
  const phaseInData = isDateInData(phaseDate(event, phase), meta);
  const days = phase === "during" ? eventDays(event).filter((d) => isDateInData(d, meta)) : [];
  const dayIndex = Math.max(0, days.indexOf(currentDate));

  return (
    <div className="p-3 space-y-3">
      <div role="tablist" aria-label="Event phase" className="grid grid-cols-3 gap-1 rounded-lg bg-slate-100 dark:bg-slate-900 p-1">
        {PHASES.map((p) => {
          const d = phaseDate(event, p.id);
          const ok = isDateInData(d, meta);
          return (
            <button
              key={p.id}
              type="button"
              role="tab"
              aria-selected={phase === p.id}
              onClick={() => onPhase(p.id)}
              title={ok ? undefined : "This date is not in the data on this site yet"}
              className={`rounded-md px-2 py-1.5 text-xs ${
                phase === p.id
                  ? "bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow font-semibold"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              } ${ok ? "" : "opacity-60"}`}
            >
              <span className="block">{p.label}</span>
              <span className="block text-[10px] font-normal">{formatDate(d)}</span>
            </button>
          );
        })}
      </div>

      {!phaseInData && (
        <p className="text-[11px] rounded-md bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-900 dark:text-amber-100 px-2 py-1.5">
          {formatDate(phaseDate(event, phase))} is not in the data on this site yet, so the map stays on{" "}
          {formatDate(currentDate)}. The map has moved to the event area.
        </p>
      )}

      {phase === "during" && days.length > 1 && (
        <div className="space-y-1">
          <label htmlFor="event-day" className="text-[11px] text-slate-600 dark:text-slate-400">
            Day {dayIndex + 1} of {days.length}: {formatDate(days[dayIndex])}
          </label>
          <input
            id="event-day"
            type="range"
            min={0}
            max={days.length - 1}
            value={dayIndex}
            onChange={(e) => onSetDate(days[Number(e.target.value)])}
            className="w-full accent-cyan-600"
          />
        </div>
      )}

      <p className="text-xs text-slate-700 dark:text-slate-300">{event.summary}</p>

      <a
        href={event.source_url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-[11px] text-cyan-700 dark:text-cyan-400 underline"
      >
        Source: {event.source_name} <ExternalLink className="h-3 w-3" />
      </a>

      {onExplain && (
        <button
          type="button"
          onClick={() => onExplain(event)}
          className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium px-3 py-2"
        >
          <Sparkles className="h-3.5 w-3.5" /> Explain what changed
        </button>
      )}
    </div>
  );
}

export default TimeMachine;
