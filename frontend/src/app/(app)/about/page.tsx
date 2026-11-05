"use client";

import React from "react";
import Link from "next/link";
import {
  Waves,
  Database,
  Cpu,
  ShieldAlert,
  ExternalLink,
  Layers,
  Calendar,
  Radio,
} from "lucide-react";
import { useMeta } from "../../../lib/api/hooks";
import { Loading } from "../../../components/common/Loading";
import { ErrorState } from "../../../components/common/ErrorState";
import { PageHero, heroActionClass } from "../../../components/common/PageHero";
import { manrope } from "../../../components/landing/fonts";
import { formatDepth } from "../../../lib/format";

export default function AboutPage() {
  const { data: meta, isLoading, error, refetch } = useMeta();

  if (isLoading) {
    return <Loading fullScreen label="Loading system details & documentation..." />;
  }

  if (error || !meta) {
    return (
      <div className="flex min-h-[60vh] flex-1 items-center justify-center bg-neutral-100 p-6 dark:bg-[#0a0c0d]">
        <ErrorState
          title="Could not connect to model metadata service"
          message={error instanceof Error ? error.message : "Metadata unavailable"}
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  const dataSources = [
    {
      name: "DUACS SLA / ADT",
      param: "Sea Level Anomaly & Absolute Dynamic Topography",
      provider: "Copernicus Marine Service",
      link: "https://marine.copernicus.eu",
      citation: "E.U. Copernicus Marine Service Information. SEALEVEL_GLO_PHY_L4_MY_008_047.",
    },
    {
      name: "OSTIA SST",
      param: "Sea Surface Temperature (Foundation)",
      provider: "Copernicus Marine Service / UK Met Office",
      link: "https://marine.copernicus.eu",
      citation: "Operational Sea Surface Temperature and Ice Analysis. SST_GLO_SST_L4_REP_OBSERVATIONS_010_001.",
    },
    {
      name: "MULTIOBS SSS",
      param: "Sea Surface Salinity",
      provider: "Copernicus Marine Service",
      link: "https://marine.copernicus.eu",
      citation: "Global Multi Observation SSS. MULTIOBS_GLO_PHY_SSS_L4_MYNRT_015_013.",
    },
    {
      name: "OSCAR Surface Currents",
      param: "Ocean Surface Current Analyses Real-time (u, v)",
      provider: "NASA PO.DAAC / ESR",
      link: "https://podaac.jpl.nasa.gov",
      citation: "ESR. OSCAR Third Degree Ocean Surface Velocity. PO.DAAC, CA, USA.",
    },
    {
      name: "CCMP Surface Winds",
      param: "Cross-Calibrated Multi-Platform Wind Vector (u10, v10)",
      provider: "Copernicus Marine / Remote Sensing Systems",
      link: "https://www.remss.com",
      citation: "Remote Sensing Systems CCMP V3.0 Ocean Vector Wind.",
    },
    {
      name: "ERA5 Reanalysis",
      param: "Atmospheric Surface Heat Fluxes",
      provider: "Copernicus Climate Change Service (CDS) / ECMWF",
      link: "https://cds.climate.copernicus.eu",
      citation: "Hersbach et al. (2020). The ERA5 global reanalysis. Q.J.R. Meteorol. Soc.",
    },
    {
      name: "GEBCO Bathymetry",
      param: "Global Ocean Sea Floor Topography & Land Mask",
      provider: "BODC / GEBCO",
      link: "https://www.gebco.net",
      citation: "GEBCO Compilation Group (2023) GEBCO 2023 Grid (doi:10.5285/f98b0f38-d45c-4d32-e053-6c86abc0f30e).",
    },
    {
      name: "WOA23 Climatology",
      param: "World Ocean Atlas 2023 In-Situ Temperature Baseline",
      provider: "NOAA NCEI",
      link: "https://www.ncei.noaa.gov/products/world-ocean-atlas",
      citation: "Locarnini et al. (2023). World Ocean Atlas 2023, Volume 1: Temperature. NOAA Atlas NESDIS.",
    },
    {
      name: "Climate Indices",
      param: "DMI (Indian Ocean Dipole) & Niño 3.4 (ENSO)",
      provider: "NOAA CPC & Australian Bureau of Meteorology",
      link: "https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/",
      citation: "NOAA Climate Prediction Center & Bureau of Meteorology Climate Influences.",
    },
    {
      name: "GLORYS12V1 Reanalysis",
      param: "High-Resolution Physical Ocean Reanalysis (Target)",
      provider: "Copernicus Marine Service / Mercator Ocean",
      link: "https://marine.copernicus.eu",
      citation: "Jean-Michel et al. (2021). The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis. Front. Earth Sci.",
    },
    {
      name: "GREP Multi-Reanalysis Ensemble",
      param: "Global Reanalysis Ensemble Product (Baseline)",
      provider: "Copernicus Marine Service",
      link: "https://marine.copernicus.eu",
      citation: "Storto et al. (2019). The GREP ensemble ocean reanalysis. Ocean Sci.",
    },
    {
      name: "ARMOR3D L4",
      param: "Multi-Observation Reprocessed 3D Temperature (Baseline)",
      provider: "Copernicus Marine Service",
      link: "https://marine.copernicus.eu",
      citation: "Guinehut et al. (2012). High-resolution 3-D temperature and salinity fields. Ocean Sci.",
    },
    {
      name: "ARGO In-Situ Float Array",
      param: "Autonomous Profiling CTD Floats (Independent Ground Truth)",
      provider: "Copernicus Marine Service / Coriolis / Argo GDAC",
      link: "https://argo.ucsd.edu",
      citation: "Argo (2023). Argo float data and metadata from Global Data Assembly Centre (GDAC). SEANOE.",
    },
  ];

  const card = "hud-glass rounded-[20px] p-5 sm:p-6";
  const sectionTitle =
    "flex items-center gap-2.5 text-lg font-medium tracking-tight text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif]";
  const prose = "space-y-3 text-sm leading-relaxed text-neutral-700 dark:text-neutral-300";
  const icon = "h-4 w-4 shrink-0 text-cyan-700 dark:text-cyan-300";

  const params = [
    { label: "Run ID", value: meta.run_id },
    {
      label: "Spatial Grid",
      value: `${meta.grid_resolution_deg}° (${meta.region.lat_min}°–${meta.region.lat_max}°N, ${meta.region.lon_min}°–${meta.region.lon_max}°E)`,
    },
    {
      label: "Depth Tiers",
      value: `${meta.depths_m.length} depths (${formatDepth(meta.depths_m[0])} to ${formatDepth(meta.depths_m[meta.depths_m.length - 1])})`,
    },
    { label: "Available Dates", value: `${meta.available_dates.start} to ${meta.available_dates.end}` },
  ];

  const caveats = [
    meta.caveats?.glorys && { label: "GLORYS Reanalysis Reference", tone: "text-cyan-700 dark:text-cyan-300", text: meta.caveats.glorys },
    meta.caveats?.argo && { label: "ARGO Float Independence", tone: "text-amber-700 dark:text-amber-300", text: meta.caveats.argo },
    meta.caveats?.live && { label: "Operational Near-Real-Time Mode", tone: "text-amber-700 dark:text-amber-300", text: meta.caveats.live },
    meta.synthetic &&
      meta.caveats?.synthetic && { label: "Demo Environment Notice", tone: "text-rose-700 dark:text-rose-300", text: meta.caveats.synthetic },
  ].filter(Boolean) as { label: string; tone: string; text: string }[];

  return (
    <div className={`min-h-screen flex-1 bg-neutral-100 text-neutral-900 dark:bg-[#0a0c0d] dark:text-neutral-100 ${manrope.variable}`}>
      <PageHero
        eyebrow="AI Oceanography"
        title="About OceanScope AI"
        actions={
          <>
            <Link href="/explore" className={heroActionClass}>
              <Waves aria-hidden="true" className="h-3.5 w-3.5" /> Open the Explorer
            </Link>
            <a href="#method" className={heroActionClass}>
              <Cpu aria-hidden="true" className="h-3.5 w-3.5" /> How it works
            </a>
            <Link href="/validation" className={heroActionClass}>
              Validation benchmarks
            </Link>
          </>
        }
      >
        OceanScope AI reconstructs high-resolution, daily three-dimensional subsurface ocean temperature fields across
        the North Indian Ocean using surface satellite observations, climate teleconnection indices, and deep neural
        networks.
      </PageHero>

      <div className="relative mx-auto -mt-20 max-w-6xl space-y-6 px-4 pb-16 sm:px-6 lg:px-8">
        {/* Deployment parameters, straight from /meta */}
        <section className={card}>
          <h2 className="hud-label flex items-center gap-2">
            <Radio aria-hidden="true" className="h-3.5 w-3.5" />
            <span>Active Deployment Parameters</span>
          </h2>
          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-4 lg:grid-cols-4">
            {params.map((p) => (
              <div key={p.label}>
                <dt className="text-[11px] text-neutral-500 dark:text-neutral-400">{p.label}</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums text-neutral-950 dark:text-white">{p.value}</dd>
              </div>
            ))}
          </dl>
        </section>

        <div className="grid gap-6 lg:grid-cols-[1.45fr_1fr]">
          <div className="space-y-6">
            <section className={`${card} space-y-4`}>
              <h2 className={sectionTitle}>
                <Layers aria-hidden="true" className={icon} />
                <span>The Challenge of Subsurface Ocean Observation</span>
              </h2>
              <div className={prose}>
                <p>
                  While Earth observation satellites measure surface characteristics (sea surface temperature, sea surface height, salinity, and surface winds) with high spatial and temporal coverage, satellites cannot penetrate into the ocean depths. Direct subsurface measurements rely almost exclusively on autonomous ARGO profiling floats, which drift freely and provide sparse point measurements separated by hundreds of kilometres and 10-day intervals.
                </p>
                <p>
                  In the North Indian Ocean—encompassing the Arabian Sea and the Bay of Bengal—subsurface thermal stratification dictates cyclone intensification rates, monsoon progression, and marine heatwave development. Conventional numerical reanalysis models (such as GLORYS12) assimilate observations into coupled hydrodynamic equations, but require vast supercomputing infrastructure and can lag real-time observations by weeks.
                </p>
              </div>
            </section>

            <section className={`${card} scroll-mt-24 space-y-4`} id="method">
              <h2 className={sectionTitle}>
                <Cpu aria-hidden="true" className={icon} />
                <span>Our Methodology</span>
              </h2>
              <div className={prose}>
                <p>
                  OceanScope AI implements an end-to-end deep learning framework designed to reconstruct daily three-dimensional temperature fields down to 1500 metres:
                </p>
                <ul className="space-y-2.5">
                  {[
                    ["Multi-Modal Surface Forcing", "Inputs comprise sea level anomaly (SLA), sea surface temperature (SST), sea surface salinity (SSS), vector wind stress (CCMP), ocean surface currents (OSCAR), and surface heat flux anomalies (ERA5) stacked with 7-day and 60-day lag histories."],
                    ["Climate Teleconnection & Float Conditioning", "Basin-wide climate indices (Indian Ocean Dipole DMI, ENSO Niño 3.4) are broadcast across the grid as spatial scalar channels, alongside recent ARGO profile observations."],
                    ["U-Net Architecture", "A small, highly regularized 2D U-Net processes multi-channel spatial fields to predict temperature anomalies relative to the World Ocean Atlas 2023 (WOA23) climatological mean across 15 depth tiers simultaneously."],
                    ["Staged Transfer Learning", "The network is pretrained on GLORYS12 physical reanalysis and the GREP multi-reanalysis ensemble, and subsequently fine-tuned on verified ARGO in-situ CTD profiles."],
                  ].map(([head, text]) => (
                    <li key={head} className="flex gap-3">
                      <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-600 dark:bg-cyan-300" />
                      <span>
                        <strong className="font-medium text-neutral-950 dark:text-white">{head}:</strong> {text}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          </div>

          <div className="space-y-6">
            <section className={`${card} space-y-3`}>
              <h2 className={sectionTitle}>
                <Calendar aria-hidden="true" className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-300" />
                <span>Understanding &quot;Live&quot; Mode</span>
              </h2>
              <div className={prose}>
                <p>
                  Dates classified as <strong className="font-medium text-amber-700 dark:text-amber-300">Live</strong> represent near-real-time inferences generated approximately 3 days behind real time using operational satellite feeds before official numerical reanalyses are published.
                </p>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  Because physical numerical reanalyses (such as GLORYS12) require several weeks of assimilation delay, no ground-truth reference or difference fields are available for live dates.
                </p>
              </div>
            </section>

            {caveats.length > 0 && (
              <section className={`${card} space-y-4`}>
                <h2 className={sectionTitle}>
                  <ShieldAlert aria-hidden="true" className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-300" />
                  <span>Official Scientific Caveats &amp; Limitations</span>
                </h2>
                <ul className="hud-divider divide-y">
                  {caveats.map((c) => (
                    <li key={c.label} className="space-y-1 py-3 first:pt-0 last:pb-0">
                      <span className={`block text-[10px] font-semibold uppercase tracking-wider ${c.tone}`}>{c.label}</span>
                      <p className="text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">{c.text}</p>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        </div>

        {/* Data sources */}
        <section className="space-y-4 pt-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <h2 className={sectionTitle}>
              <Database aria-hidden="true" className={icon} />
              <span>Data Sources &amp; Acknowledgements</span>
            </h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              We gratefully acknowledge the data providers whose open access initiatives make this work possible.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {dataSources.map((ds) => (
              <article key={ds.name} className="hud-glass flex flex-col justify-between gap-3 rounded-2xl p-4 text-xs">
                <div>
                  <div className="flex items-start justify-between gap-3">
                    <span className="font-medium text-neutral-950 dark:text-white">{ds.name}</span>
                    <a
                      href={ds.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex shrink-0 items-center gap-1 text-[11px] text-cyan-700 hover:underline dark:text-cyan-300"
                      aria-label={`Access ${ds.name} (opens in a new tab)`}
                    >
                      <span>Access</span>
                      <ExternalLink aria-hidden="true" className="h-3 w-3" />
                    </a>
                  </div>
                  <p className="mt-1 text-[11px] text-neutral-600 dark:text-neutral-400">{ds.param}</p>
                  <p className="mt-0.5 text-[11px] text-neutral-500">{ds.provider}</p>
                </div>
                <p className="hud-divider border-t pt-2 text-[10px] italic text-neutral-500">{ds.citation}</p>
              </article>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="hud-divider flex flex-col items-center justify-between gap-4 border-t pt-6 text-xs text-neutral-500 sm:flex-row dark:text-neutral-400">
          <div>OceanScope AI • Created by Aryan Chettri &amp; AI Pair Programming Team</div>
          <nav aria-label="More" className="flex items-center gap-2">
            <Link href="/explore" className="hud-btn">
              ← Explorer Map
            </Link>
            <Link href="/validation" className="hud-btn">
              Validation Benchmarks →
            </Link>
          </nav>
        </footer>
      </div>
    </div>
  );
}
