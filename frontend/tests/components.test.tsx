import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Banner } from "../src/components/common/Banner";
import { FieldToggle } from "../src/components/controls/FieldToggle";
import { HeadlineCard } from "../src/components/validation/HeadlineCard";

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
}

function renderWithClient(ui: React.ReactElement) {
  const testQueryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={testQueryClient}>{ui}</QueryClientProvider>
  );
}

describe("Frontend Component Verification (A7)", () => {
  describe("Banner component", () => {
    it("renders synthetic warning banner", () => {
      renderWithClient(<Banner type="synthetic" />);
      const banner = screen.getByRole("alert");
      expect(banner).toBeInTheDocument();
      expect(banner.textContent).toContain("synthetic demo data");
    });

    it("renders operational live date caveat banner", () => {
      renderWithClient(<Banner type="live" />);
      const text = screen.getByText(/Live maps are made about 3 days behind real time/i);
      expect(text).toBeInTheDocument();
    });
  });

  describe("FieldToggle on live vs validation dates", () => {
    it("enables all field options when isLive is false", () => {
      const onChange = vi.fn();
      render(<FieldToggle value="pred" onChange={onChange} isLive={false} />);

      const predBtn = screen.getByRole("button", { name: /Model prediction/i });
      const glorysBtn = screen.getByRole("button", { name: /GLORYS reanalysis/i });
      const errorBtn = screen.getByRole("button", { name: /Difference/i });

      expect(predBtn).not.toBeDisabled();
      expect(glorysBtn).not.toBeDisabled();
      expect(errorBtn).not.toBeDisabled();
    });

    it("disables GLORYS and Difference when isLive is true and displays caveat", () => {
      const onChange = vi.fn();
      render(<FieldToggle value="pred" onChange={onChange} isLive={true} />);

      const predBtn = screen.getByRole("button", { name: /Model prediction/i });
      const glorysBtn = screen.getByRole("button", { name: /GLORYS reanalysis/i });
      const errorBtn = screen.getByRole("button", { name: /Difference/i });

      expect(predBtn).not.toBeDisabled();
      expect(glorysBtn).toBeDisabled();
      expect(errorBtn).toBeDisabled();

      expect(
        screen.getByText(/GLORYS & Difference unavailable on Live dates/i)
      ).toBeInTheDocument();
    });
  });

  describe("HeadlineCard verdicts and n/a handling", () => {
    const baseHeadline = {
      criterion: "Bay of Bengal, 50–300 m",
      run_id: "test-run",
      synthetic: true,
      rows: [
        {
          period: "val",
          product: "model",
          rmse_c: 0.28,
          ci_low_c: 0.24,
          ci_high_c: 0.32,
          diff_vs_model_c: null,
          diff_ci_low_c: null,
          diff_ci_high_c: null,
          n_floats: 50,
          n_obs: 1200,
        },
        {
          period: "val",
          product: "glorys12",
          rmse_c: 0.36,
          ci_low_c: 0.31,
          ci_high_c: 0.41,
          diff_vs_model_c: 0.08,
          diff_ci_low_c: 0.03,
          diff_ci_high_c: 0.13,
          n_floats: 50,
          n_obs: 1200,
        },
      ],
    };

    it("displays 'Verdict: Met' when met is true", () => {
      const headline = { ...baseHeadline, met: true };
      render(<HeadlineCard headline={headline} reportedPeriod="val" />);

      const verdict = screen.getByTestId("headline-verdict");
      expect(verdict.textContent).toContain("Verdict: Met");
    });

    it("displays 'Verdict: Not met' when met is false", () => {
      const headline = { ...baseHeadline, met: false };
      render(<HeadlineCard headline={headline} reportedPeriod="val" />);

      const verdict = screen.getByTestId("headline-verdict");
      expect(verdict.textContent).toContain("Verdict: Not met");
    });

    it("displays 'Verdict: Not yet evaluated' when met is null", () => {
      const headline = { ...baseHeadline, met: null };
      render(<HeadlineCard headline={headline} reportedPeriod="val" />);

      const verdict = screen.getByTestId("headline-verdict");
      expect(verdict.textContent).toContain("Verdict: Not yet evaluated");
    });

    it("handles null / n/a competitor rows gracefully without crashing", () => {
      const headline = {
        criterion: "Bay of Bengal, 50–300 m",
        run_id: "test-run",
        synthetic: true,
        met: null,
        rows: [
          {
            period: "test",
            product: "model",
            rmse_c: null,
            ci_low_c: null,
            ci_high_c: null,
            diff_vs_model_c: null,
            diff_ci_low_c: null,
            diff_ci_high_c: null,
            n_floats: null,
            n_obs: null,
          },
          {
            period: "test",
            product: "grep",
            rmse_c: null,
            ci_low_c: null,
            ci_high_c: null,
            diff_vs_model_c: null,
            diff_ci_low_c: null,
            diff_ci_high_c: null,
            n_floats: null,
            n_obs: null,
          },
        ],
      };

      render(<HeadlineCard headline={headline} reportedPeriod="test" />);

      expect(screen.getByTestId("headline-card")).toBeInTheDocument();
      // Should show n/a text for values
      const naElements = screen.getAllByText(/n\/a/i);
      expect(naElements.length).toBeGreaterThan(0);
    });

    it("filters card display to the specified reportedPeriod", () => {
      const headline = {
        ...baseHeadline,
        met: true,
        rows: [
          {
            period: "val",
            product: "val-only-product",
            rmse_c: 0.3,
            ci_low_c: 0.25,
            ci_high_c: 0.35,
            diff_vs_model_c: 0.05,
            diff_ci_low_c: 0.01,
            diff_ci_high_c: 0.09,
            n_floats: 50,
            n_obs: 1000,
          },
          {
            period: "test",
            product: "test-only-product",
            rmse_c: null,
            ci_low_c: null,
            ci_high_c: null,
            diff_vs_model_c: null,
            diff_ci_low_c: null,
            diff_ci_high_c: null,
            n_floats: null,
            n_obs: null,
          },
        ],
      };

      const { rerender } = render(
        <HeadlineCard headline={headline} reportedPeriod="val" />
      );
      expect(screen.getByText("val-only-product")).toBeInTheDocument();
      expect(screen.queryByText("test-only-product")).not.toBeInTheDocument();

      rerender(<HeadlineCard headline={headline} reportedPeriod="test" />);
      expect(screen.queryByText("val-only-product")).not.toBeInTheDocument();
      expect(screen.getByText("test-only-product")).toBeInTheDocument();
    });
  });
});
