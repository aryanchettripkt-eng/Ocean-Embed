/**
 * Mock data fixtures derived from fixtures/sample_outputs (PRD_FRONTEND §2, §5).
 */

import type {
  ArgoProfilesResponse,
  ArgoResponse,
  HeadlineResponse,
  HealthResponse,
  MapResponse,
  MetaResponse,
  MetricsResponse,
  ProfileResponse,
  RegionSummaryResponse,
  TimeseriesResponse,
} from "../lib/api/client";

export const mockHealth: HealthResponse = {
  status: "ok",
  data_loaded: true,
  run_id: "synthetic_v1",
  detail: null,
};

export const mockMeta: MetaResponse = {
  run_id: "synthetic_v1",
  synthetic: true,
  dataset_version: "v1",
  model_name: "SYNTHETIC_DEMO",
  region: { lon_min: 40.0, lon_max: 100.0, lat_min: 0.0, lat_max: 30.0 },
  grid_resolution_deg: 1.0,
  grid_shape: { lat: 30, lon: 60 },
  depths_m: [0.5, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 750.0, 1000.0],
  available_dates: { start: "2020-01-01", end: "2020-01-10" },
  latest_live_date: "2020-01-10",
  splits: {
    train: ["1993-03-02", "2017-12-01"],
    val: ["2018-01-01", "2020-12-01"],
    test: ["2021-01-01", "2021-12-31"],
  },
  test_evaluated: false,
  color_ranges: [
    { depth_m: 0.5, vmin: 26.0, vmax: 29.5 },
    { depth_m: 5.0, vmin: 25.8, vmax: 29.2 },
    { depth_m: 10.0, vmin: 25.5, vmax: 29.0 },
    { depth_m: 20.0, vmin: 24.5, vmax: 28.5 },
    { depth_m: 30.0, vmin: 23.5, vmax: 28.0 },
    { depth_m: 50.0, vmin: 21.5, vmax: 27.0 },
    { depth_m: 75.0, vmin: 19.5, vmax: 25.5 },
    { depth_m: 100.0, vmin: 17.5, vmax: 24.0 },
    { depth_m: 125.0, vmin: 16.0, vmax: 22.5 },
    { depth_m: 150.0, vmin: 14.5, vmax: 21.0 },
    { depth_m: 200.0, vmin: 12.5, vmax: 18.5 },
    { depth_m: 300.0, vmin: 10.5, vmax: 15.0 },
    { depth_m: 500.0, vmin: 8.0, vmax: 11.5 },
    { depth_m: 750.0, vmin: 6.0, vmax: 9.0 },
    { depth_m: 1000.0, vmin: 4.5, vmax: 7.0 },
  ],
  error_range: { vmin: -2.0, vmax: 2.0 },
  caveats: {
    glorys:
      "Predictions are trained to reproduce the GLORYS reanalysis. GLORYS assimilates satellite altimetry and ARGO profiles, so agreement with GLORYS is not the same as agreement with the real ocean.",
    argo: "ARGO profiles are also assimilated into GLORYS, so this comparison is not fully independent.",
    live:
      "Live maps are made about 3 days behind real time from near-real-time inputs. There is no GLORYS reference for these dates yet, so they have not been checked the way validation dates have.",
    synthetic: "This deployment is serving synthetic demo data, not model results.",
  },
  chat: { enabled: false, provider_label: null },
};

// Generate 30 x 60 grid coordinates
const lats = Array.from({ length: 30 }, (_, i) => 0.5 + i);
const lons = Array.from({ length: 60 }, (_, i) => 40.5 + i);

// Grid slice
export function createMockMap(date = "2020-01-05", depth_m = 100.0, field: "pred" | "glorys" | "error" = "pred"): MapResponse {
  const isLive = date >= "2020-01-09";
  const split = isLive ? "live" : "val";

  const values: (number | null)[][] = [];
  const coverage: number[][] = [];

  for (let r = 0; r < 30; r++) {
    const rowVals: (number | null)[] = [];
    const rowCov: number[] = [];
    const lat = lats[r];

    for (let c = 0; c < 60; c++) {
      const lon = lons[c];
      // Simple synthetic land mask: India peninsula
      const isIndiaLand = lat >= 8 && lat <= 26 && lon >= 72 && lon <= 88 && (lat - 8) > (lon - 72) * 0.7;

      if (isIndiaLand) {
        rowCov.push(0); // land
        rowVals.push(null);
      } else if (r === 28 && c === 10) {
        rowCov.push(2); // insufficient inputs
        rowVals.push(null);
      } else {
        rowCov.push(1); // valid ocean
        if (isLive && field !== "pred") {
          rowVals.push(null);
        } else {
          // Temperature decreasing with depth and increasing towards equator
          const base = 28.5 - Math.log10(depth_m + 1) * 7.5 - (lat / 30.0) * 2.0;
          const val = field === "error" ? 0.08 : base;
          rowVals.push(Math.round(val * 100) / 100);
        }
      }
    }
    values.push(rowVals);
    coverage.push(rowCov);
  }

  return {
    run_id: "synthetic_v1",
    synthetic: true,
    date,
    depth_m,
    field,
    split,
    units: "degC",
    lat: lats,
    lon: lons,
    values,
    coverage,
    stats: {
      min: 17.5,
      max: 23.5,
      mean: 20.8,
      n_valid: 1450,
    },
  };
}

export function createMockProfile(lat = 12.5, lon = 65.5, date = "2020-01-05"): ProfileResponse {
  const isLive = date >= "2020-01-09";
  const depths = mockMeta.depths_m;
  const pred = depths.map((d) => Math.round((28.5 - Math.log10(d + 1) * 7.5) * 100) / 100);
  const glorys = isLive
    ? depths.map(() => null)
    : depths.map((d) => Math.round((28.4 - Math.log10(d + 1) * 7.4) * 100) / 100);
  const error = isLive
    ? depths.map(() => null)
    : depths.map((_, i) => Math.round(((pred[i] as number) - (glorys[i] as number)) * 100) / 100);

  return {
    run_id: "synthetic_v1",
    synthetic: true,
    date,
    lat,
    lon,
    split: isLive ? "live" : "val",
    coverage: 1,
    depths_m: depths,
    pred,
    glorys,
    error,
  };
}

export function createMockTimeseries(lat = 12.5, lon = 65.5, start = "2020-01-01", end = "2020-01-10"): TimeseriesResponse {
  const dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"];
  const pred = [21.1, 21.2, 21.15, 21.3, 21.25, 21.2, 21.18, 21.22, 21.25, 21.3];
  const glorys = [21.05, 21.12, 21.1, 21.22, 21.2, 21.15, 21.12, 21.18, null, null];
  const split = ["val", "val", "val", "val", "val", "val", "val", "val", "live", "live"];

  return {
    run_id: "synthetic_v1",
    synthetic: true,
    lat,
    lon,
    depth_m: 100.0,
    dates,
    pred,
    glorys,
    split,
  };
}

export const mockRegionSummary: RegionSummaryResponse = {
  run_id: "synthetic_v1",
  synthetic: true,
  date: "2020-01-05",
  depth_m: 100.0,
  field: "pred",
  mean: 20.85,
  min: 17.5,
  max: 23.5,
  std: 1.25,
  n_valid: 1450,
  n_total: 1800,
};

export const mockMetrics: MetricsResponse = {
  run_id: "synthetic_v1",
  synthetic: true,
  split: "val",
  group_by: "depth",
  rows: [
    { model: "SYNTHETIC_DEMO", split: "val", depth_m: 0.5, group: null, n: 7208, mae_c: 0.07, rmse_c: 0.07, bias_c: 0.07, skill_vs_clim_pct: 87.9 },
    { model: "SYNTHETIC_DEMO", split: "val", depth_m: 5.0, group: null, n: 7208, mae_c: 0.09, rmse_c: 0.09, bias_c: 0.09, skill_vs_clim_pct: 84.5 },
    { model: "SYNTHETIC_DEMO", split: "val", depth_m: 10.0, group: null, n: 7208, mae_c: 0.1, rmse_c: 0.1, bias_c: 0.1, skill_vs_clim_pct: 82.8 },
    { model: "SYNTHETIC_DEMO", split: "val", depth_m: 50.0, group: null, n: 7208, mae_c: 0.21, rmse_c: 0.21, bias_c: 0.21, skill_vs_clim_pct: 63.8 },
    { model: "SYNTHETIC_DEMO", split: "val", depth_m: 100.0, group: null, n: 7208, mae_c: 0.29, rmse_c: 0.29, bias_c: 0.29, skill_vs_clim_pct: 50.0 },
    { model: "SYNTHETIC_DEMO", split: "val", depth_m: "all", group: null, n: 108120, mae_c: 0.19, rmse_c: 0.2, bias_c: 0.15, skill_vs_clim_pct: 65.5 },
    { model: "climatology", split: "val", depth_m: 100.0, group: null, n: 7208, mae_c: 0.44, rmse_c: 0.58, bias_c: 0.0, skill_vs_clim_pct: 0.0 },
    { model: "climatology", split: "val", depth_m: "all", group: null, n: 108120, mae_c: 0.44, rmse_c: 0.58, bias_c: 0.0, skill_vs_clim_pct: 0.0 },
    { model: "ridge", split: "val", depth_m: 100.0, group: null, n: 7208, mae_c: 0.32, rmse_c: 0.41, bias_c: 0.02, skill_vs_clim_pct: 29.3 },
    { model: "ridge", split: "val", depth_m: "all", group: null, n: 108120, mae_c: 0.32, rmse_c: 0.41, bias_c: 0.02, skill_vs_clim_pct: 29.3 },
  ],
};

export const mockArgo: ArgoResponse = {
  run_id: "synthetic_v1",
  synthetic: true,
  n_profiles_total: 3,
  n_obs_total: 45,
  caveat: "ARGO profiles are also assimilated into GLORYS, so this comparison is not fully independent.",
  rows: [
    { depth_m: 0.5, n_profiles: 3, n_obs: 3, mae_c: 0.08, rmse_c: 0.08, bias_c: -0.08, glorys_mae_c: 0.08, glorys_rmse_c: 0.08, grep_mae_c: 0.08, grep_rmse_c: 0.08, armor3d_mae_c: 0.14, armor3d_rmse_c: 0.14 },
    { depth_m: 5.0, n_profiles: 3, n_obs: 3, mae_c: 0.07, rmse_c: 0.07, bias_c: -0.07, glorys_mae_c: 0.08, glorys_rmse_c: 0.08, grep_mae_c: 0.06, grep_rmse_c: 0.06, armor3d_mae_c: 0.14, armor3d_rmse_c: 0.14 },
    { depth_m: 10.0, n_profiles: 3, n_obs: 3, mae_c: 0.05, rmse_c: 0.05, bias_c: -0.05, glorys_mae_c: 0.08, glorys_rmse_c: 0.08, grep_mae_c: 0.04, grep_rmse_c: 0.04, armor3d_mae_c: 0.14, armor3d_rmse_c: 0.14 },
    { depth_m: 50.0, n_profiles: 3, n_obs: 3, mae_c: 0.07, rmse_c: 0.07, bias_c: 0.07, glorys_mae_c: 0.04, glorys_rmse_c: 0.04, grep_mae_c: 0.12, grep_rmse_c: 0.12, armor3d_mae_c: 0.03, armor3d_rmse_c: 0.03 },
    { depth_m: 100.0, n_profiles: 3, n_obs: 3, mae_c: 0.18, rmse_c: 0.18, bias_c: 0.18, glorys_mae_c: 0.03, glorys_rmse_c: 0.03, grep_mae_c: 0.18, grep_rmse_c: 0.18, armor3d_mae_c: 0.14, armor3d_rmse_c: 0.14 },
  ],
};

export const mockArgoProfiles: ArgoProfilesResponse = {
  run_id: "synthetic_v1",
  synthetic: true,
  profiles: [
    { platform_number: "2902123", cycle: 14, time: "2020-01-03T06:00:00Z", lat: 12.5, lon: 65.5, n_obs: 15 },
    { platform_number: "2902124", cycle: 8, time: "2020-01-05T12:00:00Z", lat: 8.5, lon: 85.5, n_obs: 15 },
    { platform_number: "2902125", cycle: 22, time: "2020-01-07T18:00:00Z", lat: 4.5, lon: 75.5, n_obs: 15 },
  ],
};

export const mockHeadline: HeadlineResponse = {
  run_id: "synthetic_v1",
  synthetic: true,
  criterion:
    "Bay of Bengal, 50–300 m: model RMSE vs held-out ARGO floats lower than GLORYS12, the GREP ensemble mean and ARMOR3D (95% float-bootstrap interval of the difference below zero).",
  met: true,
  rows: [
    { period: "val", product: "model", n_floats: 3, n_obs: 15, rmse_c: 0.28, ci_low_c: 0.24, ci_high_c: 0.32, diff_vs_model_c: 0.0, diff_ci_low_c: 0.0, diff_ci_high_c: 0.0 },
    { period: "val", product: "glorys12", n_floats: 3, n_obs: 15, rmse_c: 0.36, ci_low_c: 0.31, ci_high_c: 0.41, diff_vs_model_c: -0.08, diff_ci_low_c: -0.14, diff_ci_high_c: -0.02 },
    { period: "val", product: "grep", n_floats: 3, n_obs: 15, rmse_c: 0.39, ci_low_c: 0.34, ci_high_c: 0.45, diff_vs_model_c: -0.11, diff_ci_low_c: -0.17, diff_ci_high_c: -0.04 },
    { period: "val", product: "armor3d", n_floats: 3, n_obs: 15, rmse_c: 0.43, ci_low_c: 0.37, ci_high_c: 0.49, diff_vs_model_c: -0.15, diff_ci_low_c: -0.21, diff_ci_high_c: -0.06 },
  ],
};
