import "@testing-library/jest-dom/vitest";
import React, { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { DepthRail, depthToRailRatio, nearestDepthIndex, nearestLevelIndex } from "../src/components/map3d/DepthRail";
import { Collapse, Presence } from "../src/components/common/Motion";

// Real model levels (deployed backend), not invented: 15 depths from 0.5 m to 1062 m
const DEPTHS = [0.494, 5.078, 9.573, 18.496, 29.445, 47.374, 65.807, 92.326, 130.666, 186.126, 222.475, 266.04, 380.213, 643.567, 1062.44];

function Harness({ initial = 92.326 }: { initial?: number }) {
  const [depth, setDepth] = useState(initial);
  return <DepthRail availableDepths={DEPTHS} selectedDepth={depth} onChangeDepth={setDepth} />;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("DepthRail", () => {
  it("maps rail positions to the nearest model level on the sqrt(depth) scale", () => {
    const max = DEPTHS[DEPTHS.length - 1];
    expect(depthToRailRatio(max, max)).toBe(1);
    expect(depthToRailRatio(0, max)).toBe(0);
    DEPTHS.forEach((d, i) => expect(nearestLevelIndex(depthToRailRatio(d, max), DEPTHS, max)).toBe(i));
    // A depth that is not exactly a level (e.g. the pre-metadata default of 100 m) still has a position
    expect(nearestDepthIndex(100, DEPTHS)).toBe(7);
  });

  it("selects any level by clicking the rail, not only the labelled ticks", () => {
    const onChange = vi.fn();
    render(<DepthRail availableDepths={DEPTHS} selectedDepth={92.326} onChangeDepth={onChange} />);
    const area = screen.getByRole("slider", { name: /Ocean Depth Rail/i }).querySelector(".cursor-ns-resize") as HTMLElement;
    vi.spyOn(area, "getBoundingClientRect").mockReturnValue({ top: 0, height: 200, left: 0, width: 100, bottom: 200, right: 100, x: 0, y: 0, toJSON: () => ({}) });

    // 266 m is not one of the labelled ticks; click at its rail position
    const y = depthToRailRatio(266.04, 1062.44) * 200;
    fireEvent.pointerDown(area, { clientY: y, button: 0, pointerId: 1 });
    expect(onChange).toHaveBeenCalledWith(266.04);
  });

  it("drags through levels and steps with the keyboard from a depth that is not exactly a level", () => {
    render(<Harness initial={100} />);
    const rail = screen.getByRole("slider", { name: /Ocean Depth Rail/i });
    fireEvent.keyDown(rail, { key: "ArrowDown" });
    expect(rail).toHaveAttribute("aria-valuenow", "130.666");
    fireEvent.keyDown(rail, { key: "Home" });
    expect(rail).toHaveAttribute("aria-valuenow", "0.494");
    fireEvent.keyDown(rail, { key: "End" });
    expect(rail).toHaveAttribute("aria-valuenow", "1062.44");
  });

  it("moves one level per wheel step instead of one level per event", () => {
    render(<Harness />);
    const rail = screen.getByRole("slider", { name: /Ocean Depth Rail/i });
    // A trackpad swipe: many small deltas adding up to just over one step
    for (let i = 0; i < 7; i++) fireEvent.wheel(rail, { deltaY: 10 });
    expect(rail).toHaveAttribute("aria-valuenow", "130.666");
    // The wheel event is cancelled so the page/map doesn't scroll under the rail
    const ev = new WheelEvent("wheel", { deltaY: 80, cancelable: true });
    act(() => {
      rail.dispatchEvent(ev);
    });
    expect(ev.defaultPrevented).toBe(true);
  });
});

describe("Presence and Collapse", () => {
  it("animates a panel out, keeping its last content until the exit finishes", async () => {
    function Panel({ label }: { label: string | null }) {
      return (
        <Presence show={label !== null}>
          <p>{label ?? "cleared"}</p>
        </Presence>
      );
    }
    const { rerender } = render(<Panel label="15.125°N" />);
    expect(screen.getByText("15.125°N")).toBeInTheDocument();

    rerender(<Panel label={null} />);
    // Still showing the last content (not "cleared") while it animates out
    expect(screen.getByText("15.125°N")).toBeInTheDocument();
    expect(screen.queryByText("cleared")).not.toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText("15.125°N")).not.toBeInTheDocument());
  });

  it("hides collapsed content from the tab order once closed", async () => {
    const { rerender } = render(
      <Collapse open id="body">
        <button type="button">Inside</button>
      </Collapse>
    );
    const body = document.getElementById("body")!;
    expect(body.hidden).toBe(false);
    rerender(
      <Collapse open={false} id="body">
        <button type="button">Inside</button>
      </Collapse>
    );
    await waitFor(() => expect(body.hidden).toBe(true));
    await act(async () => {
      rerender(
        <Collapse open id="body">
          <button type="button">Inside</button>
        </Collapse>
      );
    });
    expect(body.hidden).toBe(false);
  });
});
