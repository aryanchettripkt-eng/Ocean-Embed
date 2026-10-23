import React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Landing } from "../../src/components/landing/Landing";
import { DepthRuler } from "../../src/components/landing/DepthRuler";
import { ExploreLink } from "../../src/components/landing/ExploreLink";
import { HowItWorks } from "../../src/components/landing/HowItWorks";
import { LandingFooter } from "../../src/components/landing/LandingFooter";
import { Banner } from "../../src/components/common/Banner";
import * as hooks from "../../src/lib/api/hooks";
import type { MetaResponse } from "../../src/lib/api/client";

// Mock next/navigation
const mockPush = vi.fn();
const mockPrefetch = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    prefetch: mockPrefetch,
  }),
  usePathname: () => "/welcome",
}));

// Mock next/font/google for Vitest environment
vi.mock("next/font/google", () => ({
  Pixelify_Sans: () => ({
    variable: "--font-pixelify",
    className: "font-pixelify",
  }),
  IBM_Plex_Mono: () => ({
    variable: "--font-mono",
    className: "font-mono",
  }),
}));

// Mock next/image to render standard img without React boolean attribute warnings
vi.mock("next/image", () => ({
  __esModule: true,
  default: ({
    fill: _fill,
    priority: _priority,
    fetchPriority: _fetchPriority,
    ...props
  }: React.ImgHTMLAttributes<HTMLImageElement> & {
    fill?: boolean;
    priority?: boolean;
    fetchPriority?: string;
  }) => {
    // eslint-disable-next-line @next/next/no-img-element
    return <img {...props} alt={props.alt || ""} />;
  },
}));

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
}

function renderWithClient(ui: React.ReactElement, client = createTestQueryClient()) {
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  );
}

const mockMeta: MetaResponse = {
  run_id: "test-run",
  model_name: "test-model",
  dataset_version: "1.0",
  synthetic: true,
  test_evaluated: false,
  grid_resolution_deg: 0.25,
  grid_shape: { lat: 160, lon: 280 },
  region: {
    lat_min: -10,
    lat_max: 30,
    lon_min: 40,
    lon_max: 110,
  },
  depths_m: [0.5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
  available_dates: { start: "2020-01-01", end: "2020-01-10" },
  splits: {
    train: ["1993", "2017"],
    val: ["2018", "2020"],
    test: ["2021", "2021"],
  },
  latest_live_date: null,
  color_ranges: [],
  error_range: { vmin: -2, vmax: 2 },
  caveats: {
    synthetic: "This deployment is serving synthetic demo data, not model results.",
    live: "Live caveat message",
    argo: "ARGO caveat message",
    glorys: "GLORYS caveat message",
  },
  chat: { enabled: false, provider_label: "test" },
};

describe("Landing Page Component Suite (§10)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders H1 text and tagline on first paint", () => {
    vi.spyOn(hooks, "useMeta").mockReturnValue({
      data: mockMeta,
      isLoading: false,
    } as ReturnType<typeof hooks.useMeta>);

    renderWithClient(<Landing />);

    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading).toHaveTextContent("OCEANSCOPE AI");

    expect(screen.getByText("BENEATH THE SURFACE")).toBeInTheDocument();
    expect(
      screen.getByText("A quieter way to see the hidden temperature of the ocean.")
    ).toBeInTheDocument();
  });

  it("CTA is an <a> tag with explorer href", () => {
    renderWithClient(<ExploreLink href="/" />);

    const cta = screen.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    expect(cta).toBeInTheDocument();
    expect(cta).toHaveAttribute("href", "/");
  });

  it("nav links have real hrefs in both desktop and mobile navigation", () => {
    vi.spyOn(hooks, "useMeta").mockReturnValue({
      data: mockMeta,
      isLoading: false,
    } as ReturnType<typeof hooks.useMeta>);

    renderWithClient(<Landing />);

    const nav = screen.getByRole("navigation", { name: /Landing navigation/i });
    expect(nav).toBeInTheDocument();

    const homeLinks = screen.getAllByRole("link", { name: "Home" });
    const exploreLinks = screen.getAllByRole("link", { name: "Explore" });
    const dataLinks = screen.getAllByRole("link", { name: "Data" });
    const learnLinks = screen.getAllByRole("link", { name: "Learn" });
    const aboutLinks = screen.getAllByRole("link", { name: "About" });

    expect(homeLinks[0]).toHaveAttribute("href", "/welcome");
    expect(exploreLinks[0]).toHaveAttribute("href", "/");
    expect(dataLinks[0]).toHaveAttribute("href", "/validation");
    expect(learnLinks[0]).toHaveAttribute("href", "/about#method");
    expect(aboutLinks[0]).toHaveAttribute("href", "/about");
  });

  it("DepthRuler renders labels from mocked meta.depths_m and none while loading", () => {
    // 1. While loading: ticks rendered with NO labels
    const mockUseMeta = vi.spyOn(hooks, "useMeta").mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof hooks.useMeta>);

    const { rerender } = renderWithClient(<DepthRuler />);

    expect(screen.queryByTestId("depth-tick-label")).toBeNull();

    // 2. When loaded: labels rendered from meta.depths_m
    mockUseMeta.mockReturnValue({
      data: mockMeta,
      isLoading: false,
    } as ReturnType<typeof hooks.useMeta>);

    rerender(
      <QueryClientProvider client={createTestQueryClient()}>
        <DepthRuler />
      </QueryClientProvider>
    );

    const labels = screen.getAllByTestId("depth-tick-label");
    expect(labels.length).toBeGreaterThan(0);
    // Nearest depths: 0.5 m, 100 m, 300 m, 500 m, 1000 m
    const labelTexts = labels.map((el) => el.textContent);
    expect(labelTexts).toContain("0.5 m");
    expect(labelTexts).toContain("100 m");
    expect(labelTexts).toContain("300 m");
    expect(labelTexts).toContain("500 m");
    expect(labelTexts).toContain("1000 m");
  });

  it("synthetic banner appears when synthetic: true and not when false", () => {
    // 1. When synthetic is true
    const mockUseMeta = vi.spyOn(hooks, "useMeta").mockReturnValue({
      data: { ...mockMeta, synthetic: true },
      isLoading: false,
    } as ReturnType<typeof hooks.useMeta>);

    const { rerender } = renderWithClient(<Banner type="synthetic" />);
    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(alert.textContent).toContain("synthetic demo data");

    // 2. When synthetic is false
    mockUseMeta.mockReturnValue({
      data: { ...mockMeta, synthetic: false },
      isLoading: false,
    } as ReturnType<typeof hooks.useMeta>);

    rerender(
      <QueryClientProvider client={createTestQueryClient()}>
        <Banner type="synthetic" />
      </QueryClientProvider>
    );

    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("ExploreLink skips the overlay when prefers-reduced-motion matches", () => {
    // Mock matchMedia to match prefers-reduced-motion
    window.matchMedia = vi.fn().mockImplementation((query) => ({
      matches: query.includes("prefers-reduced-motion: reduce"),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));

    // Mock document.startViewTransition exists
    (document as unknown as { startViewTransition?: unknown }).startViewTransition = vi.fn();

    const dispatchSpy = vi.spyOn(window, "dispatchEvent");

    renderWithClient(<ExploreLink href="/" />);
    const cta = screen.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    fireEvent.click(cta);

    // Should push directly and not dispatch dive overlay event
    expect(mockPush).toHaveBeenCalledWith("/");
    expect(dispatchSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "oceanscope:dive" })
    );
  });

  it("ExploreLink skips the overlay when startViewTransition is missing", () => {
    // prefers-reduced-motion is false
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));

    // delete startViewTransition from document
    delete (document as unknown as { startViewTransition?: unknown }).startViewTransition;

    const dispatchSpy = vi.spyOn(window, "dispatchEvent");

    renderWithClient(<ExploreLink href="/" />);
    const cta = screen.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    fireEvent.click(cta);

    // Should push directly without overlay
    expect(mockPush).toHaveBeenCalledWith("/");
    expect(dispatchSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "oceanscope:dive" })
    );
  });

  it("HowItWorks renders 3 step cards without numbers typed in copy", () => {
    render(<HowItWorks />);
    expect(screen.getByText("Satellites see the surface")).toBeInTheDocument();
    expect(screen.getByText("An AI model learns the ocean below")).toBeInTheDocument();
    expect(screen.getByText("You explore 15 depths, every day")).toBeInTheDocument();

    const teaser = screen.getByText(/Explore model validation and accuracy metrics/i);
    expect(teaser.closest("a")).toHaveAttribute("href", "/validation");
  });

  it("LandingFooter renders required data credits and links", () => {
    render(<LandingFooter />);
    const footer = screen.getByRole("contentinfo");
    expect(footer).toBeInTheDocument();
    expect(footer.textContent).toContain("Copernicus Marine Service");
    expect(footer.textContent).toContain("ARGO float");
  });
});
