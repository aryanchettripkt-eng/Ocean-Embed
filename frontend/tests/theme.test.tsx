import React from "react";
import { beforeEach, afterEach, describe, it, expect, vi } from "vitest";
import {
  render,
  screen,
  fireEvent,
  act,
  cleanup,
} from "@testing-library/react";
import { useTheme } from "../src/lib/use-theme";
function Control({ name }: { name: string }) {
  const { theme, toggleTheme } = useTheme();
  return (
    <button onClick={toggleTheme}>
      {name}: {theme}
    </button>
  );
}
describe("Shared landing and Explorer theme", () => {
  let listener: (() => void) | undefined;
  let systemDark = false;
  const originalStorage = Object.getOwnPropertyDescriptor(
    window,
    "localStorage",
  );
  beforeEach(() => {
    const values = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => {
          values.set(key, String(value));
        },
        removeItem: (key: string) => values.delete(key),
        clear: () => values.clear(),
        key: (index: number) => [...values.keys()][index] ?? null,
        get length() {
          return values.size;
        },
      },
    });
    document.documentElement.className = "";
    listener = undefined;
    systemDark = false;
    vi.spyOn(window, "matchMedia").mockImplementation((query) => ({
      get matches() {
        return systemDark;
      },
      media: query,
      onchange: null,
      addEventListener: (
        _event: string,
        callback: EventListenerOrEventListenerObject,
      ) => {
        listener = callback as () => void;
      },
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    if (originalStorage)
      Object.defineProperty(window, "localStorage", originalStorage);
    else Reflect.deleteProperty(window, "localStorage");
  });
  it("follows the system until a preference is explicitly saved", () => {
    render(<Control name="landing" />);
    expect(document.documentElement).toHaveClass("light");
    act(() => {
      systemDark = true;
      listener?.();
    });
    expect(document.documentElement).toHaveClass("dark");
    fireEvent.click(screen.getByRole("button"));
    expect(window.localStorage.getItem("theme")).toBe("light");
    act(() => {
      systemDark = true;
      listener?.();
    });
    expect(document.documentElement).toHaveClass("light");
  });
  it("uses a saved preference and synchronizes mounted controls", () => {
    window.localStorage.setItem("theme", "dark");
    render(
      <>
        <Control name="landing" />
        <Control name="header" />
      </>,
    );
    expect(screen.getByText("header: dark")).toBeInTheDocument();
    fireEvent.click(screen.getByText("landing: dark"));
    expect(screen.getByText("header: light")).toBeInTheDocument();
    expect(document.documentElement).not.toHaveClass("dark");
  });
  it("still toggles and synchronizes when browser storage is blocked", () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new Error("Blocked");
    });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("Blocked");
    });
    render(
      <>
        <Control name="landing" />
        <Control name="header" />
      </>,
    );
    fireEvent.click(screen.getByText("landing: light"));
    expect(screen.getByText("header: dark")).toBeInTheDocument();
    expect(document.documentElement).toHaveClass("dark");
  });
});
