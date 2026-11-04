// Served dates can have a gap: live mode appends recent days after the historical ones
// (backend /meta `date_ranges`). These helpers walk only the days the store holds, so the
// date slider, step buttons and play loop skip the gap instead of landing on missing days.

export interface DayRange {
  start: string;
  end: string;
}

export function dateToDays(dateStr: string): number {
  return Math.floor(new Date(dateStr + "T00:00:00Z").getTime() / 86400000);
}

export function daysToDate(days: number): string {
  return new Date(days * 86400000).toISOString().slice(0, 10);
}

export class ServedDays {
  private readonly segs: { s: number; e: number; offset: number }[];
  readonly count: number;

  /** `ranges` from /meta; falls back to one range [start, end] when missing or empty. */
  constructor(start: string, end: string, ranges?: DayRange[] | null) {
    const lo = dateToDays(start);
    const hi = dateToDays(end);
    const raw = ranges && ranges.length > 0 ? ranges : [{ start, end }];
    const segs = raw
      .map((r) => ({ s: Math.max(lo, dateToDays(r.start)), e: Math.min(hi, dateToDays(r.end)) }))
      .filter((r) => r.s <= r.e)
      .sort((a, b) => a.s - b.s);
    let offset = 0;
    this.segs = segs.map((r) => {
      const seg = { ...r, offset };
      offset += r.e - r.s + 1;
      return seg;
    });
    if (this.segs.length === 0) {
      this.segs = [{ s: lo, e: hi, offset: 0 }];
      offset = hi - lo + 1;
    }
    this.count = offset;
  }

  get first(): string {
    return daysToDate(this.segs[0].s);
  }

  get last(): string {
    return daysToDate(this.segs[this.segs.length - 1].e);
  }

  has(date: string): boolean {
    const d = dateToDays(date);
    return this.segs.some((r) => d >= r.s && d <= r.e);
  }

  /** Position on the gap-free axis; a day in a gap maps to the next served day. */
  indexOf(date: string): number {
    const d = dateToDays(date);
    for (const r of this.segs) {
      if (d < r.s) return r.offset;
      if (d <= r.e) return r.offset + (d - r.s);
    }
    return this.count - 1;
  }

  at(index: number): string {
    const i = Math.max(0, Math.min(this.count - 1, Math.round(index)));
    for (const r of this.segs) {
      if (i < r.offset + (r.e - r.s + 1)) return daysToDate(r.s + (i - r.offset));
    }
    return this.last;
  }

  /** The served day `step` days away (±1), jumping over gaps; null past either end. */
  step(date: string, step: 1 | -1): string | null {
    const d = dateToDays(date);
    if (step === 1) {
      for (const r of this.segs) {
        if (d + 1 <= r.e) return daysToDate(Math.max(d + 1, r.s));
      }
      return null;
    }
    for (let k = this.segs.length - 1; k >= 0; k--) {
      const r = this.segs[k];
      if (d - 1 >= r.s) return daysToDate(Math.min(d - 1, r.e));
    }
    return null;
  }

  /** The served day nearest a typed date (the next one when it falls in a gap). */
  snap(date: string): string {
    return this.has(date) ? date : this.at(this.indexOf(date));
  }
}
