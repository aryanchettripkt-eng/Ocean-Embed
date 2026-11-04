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
      className="hud-glass flex max-h-[60vh] w-full flex-col overflow-hidden rounded-[20px] sm:w-[380px]"
    >
      <header className="hud-divider flex shrink-0 items-center justify-between gap-2 border-b px-4 py-3">
        <div className="flex items-center gap-2 min-w-0">
          {active && (
            <button type="button" onClick={onBack} aria-label="Back to events" className="hud-icon-btn">
              <ArrowLeft aria-hidden="true" className="h-4 w-4" />
            </button>
          )}
          <CalendarRange aria-hidden="true" className="h-4 w-4 shrink-0 text-neutral-500" />
          <h2 className="truncate text-sm font-semibold text-neutral-950 dark:text-white">
            {active ? active.event.title : "Ocean events"}
          </h2>
        </div>
        <button type="button" onClick={onClose} aria-label="Close events" className="hud-icon-btn">
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </header>

      {!active && (
        <div className="min-h-0 space-y-2 overflow-y-auto overscroll-contain p-3">
          {isLoading && <p className="text-xs text-neutral-500">Loading events…</p>}
          {isError && <p className="text-xs text-red-600 dark:text-red-300">Couldn&apos;t load events. Try again later.</p>}
          {!isLoading && !isError && events.length === 0 && (
            <p className="text-xs text-neutral-500">No events are available yet.</p>
          )}
          {events.map((e) => (
            <button
              key={e.id}
              type="button"
              onClick={() => onOpen(e, "before", { pickDepth: true })}
              className="block w-full rounded-2xl bg-black/[0.03] px-3.5 py-2.5 text-left ring-1 ring-inset ring-black/[0.05] transition-colors hover:bg-black/[0.06] dark:bg-white/[0.04] dark:ring-white/[0.07] dark:hover:bg-white/[0.08]"
            >
              <p className="text-sm font-medium text-neutral-950 dark:text-white">{e.title}</p>
              <p className="mt-0.5 text-[11px] text-neutral-500 dark:text-neutral-400">
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
    <div className="min-h-0 space-y-3 overflow-y-auto overscroll-contain p-4">
      <div role="tablist" aria-label="Event phase" className="grid grid-cols-3 gap-0.5 rounded-2xl bg-black/[0.04] p-0.5 dark:bg-white/[0.05]">
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
              className={`rounded-[14px] px-2 py-1.5 text-xs transition-colors ${
                phase === p.id
                  ? "bg-neutral-900 font-semibold text-white shadow-sm dark:bg-white dark:text-neutral-950"
                  : "text-neutral-600 hover:text-neutral-950 dark:text-neutral-400 dark:hover:text-white"
              } ${ok ? "" : "opacity-60"}`}
            >
              <span className="block">{p.label}</span>
              <span className="block text-[10px] font-normal opacity-70">{formatDate(d)}</span>
            </button>
          );
        })}
      </div>

      {!phaseInData && (
        <p className="rounded-xl bg-amber-500/10 px-3 py-2 text-[11px] text-amber-900 ring-1 ring-inset ring-amber-500/25 dark:text-amber-100">
          {formatDate(phaseDate(event, phase))} is not in the data on this site yet, so the map stays on{" "}
          {formatDate(currentDate)}. The map has moved to the event area.
        </p>
      )}

      {phase === "during" && days.length > 1 && (
        <div className="space-y-1">
          <label htmlFor="event-day" className="hud-label">
            Day {dayIndex + 1} of {days.length}: {formatDate(days[dayIndex])}
          </label>
          <input
            id="event-day"
            type="range"
            min={0}
            max={days.length - 1}
            value={dayIndex}
            onChange={(e) => onSetDate(days[Number(e.target.value)])}
            className="hud-range"
            style={{ "--fill": `${days.length > 1 ? (dayIndex / (days.length - 1)) * 100 : 0}%` } as React.CSSProperties}
          />
        </div>
      )}

      <p className="text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">{event.summary}</p>

      <a
        href={event.source_url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-[11px] text-cyan-700 underline underline-offset-2 dark:text-cyan-300"
      >
        Source: {event.source_name} <ExternalLink aria-hidden="true" className="h-3 w-3" />
      </a>

      {onExplain && (
        <button
          type="button"
          onClick={() => onExplain(event)}
          className="hud-btn w-full bg-cyan-400 font-semibold text-neutral-950 hover:bg-cyan-300 hover:text-neutral-950 dark:text-neutral-950 dark:hover:bg-cyan-300 dark:hover:text-neutral-950"
        >
          <Sparkles aria-hidden="true" className="h-3.5 w-3.5" /> Explain what changed
        </button>
      )}
    </div>
  );
}

export default TimeMachine;
