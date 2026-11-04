import "@testing-library/jest-dom/vitest";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";

import {
  arrivalCovered,
  arrivalMapReady,
  arrivalRevealed,
  beginArrival,
  finishArrival,
  getArrivalPhase,
  isArrivalPending,
} from "../src/lib/arrival";
import { ArrivalOverlay } from "../src/components/transition/ArrivalOverlay";

afterEach(() => {
  cleanup();
  finishArrival();
  vi.useRealTimers();
});

describe("arrival state machine", () => {
  it("runs sweep → cover (navigate) → reveal → idle and hides the HUD until finished", () => {
    const navigate = vi.fn();
    expect(beginArrival(navigate)).toBe(true);
    expect(getArrivalPhase()).toBe("sweeping");
    expect(isArrivalPending()).toBe(true);
    // A second click while running is ignored
    expect(beginArrival(vi.fn())).toBe(false);

    arrivalCovered();
    expect(getArrivalPhase()).toBe("covering");
    expect(navigate).toHaveBeenCalledTimes(1);

    arrivalMapReady();
    expect(getArrivalPhase()).toBe("revealing");
    arrivalRevealed();
    expect(getArrivalPhase()).toBe("idle");
    // The Explorer's entrance is still pending until it says so
    expect(isArrivalPending()).toBe(true);
    finishArrival();
    expect(isArrivalPending()).toBe(false);
  });
});

describe("ArrivalOverlay", () => {
  it("covers the screen, navigates when covered and fades out when the map is ready", async () => {
    const navigate = vi.fn();
    render(<ArrivalOverlay />);
    expect(screen.queryByRole("status", { name: /Entering the ocean explorer/i })).not.toBeInTheDocument();

    act(() => {
      beginArrival(navigate);
    });
    expect(screen.getByRole("status", { name: /Entering the ocean explorer/i })).toBeInTheDocument();
    // The surface sweep completes, then the route change is requested
    await waitFor(() => expect(navigate).toHaveBeenCalledTimes(1), { timeout: 2000 });
    expect(getArrivalPhase()).toBe("covering");

    act(() => arrivalMapReady());
    await waitFor(() => expect(getArrivalPhase()).toBe("idle"), { timeout: 2000 });
    expect(screen.queryByRole("status", { name: /Entering the ocean explorer/i })).not.toBeInTheDocument();
  });

  it("never traps the user: gives up after 15 s and shows the HUD", async () => {
    vi.useFakeTimers();
    render(<ArrivalOverlay />);
    act(() => {
      beginArrival(vi.fn());
      arrivalCovered();
    });
    act(() => {
      vi.advanceTimersByTime(4000);
    });
    expect(screen.getByText(/Waking up the ocean server/)).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(11_000);
    });
    expect(isArrivalPending()).toBe(false);
    expect(["revealing", "idle"]).toContain(getArrivalPhase());
  });
});
