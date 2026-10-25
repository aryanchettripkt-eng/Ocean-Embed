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
import { Banner } from "../../../components/common/Banner";
import { Loading } from "../../../components/common/Loading";
import { formatDepth } from "../../../lib/format";

export default function AboutPage() {
  const { data: meta, isLoading, error, refetch } = useMeta();

  if (isLoading) {
    return <Loading fullScreen label="Loading system details & documentation..." />;
  }

  if (error || !meta) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center p-6 text-center">
        <div className="space-y-3">
          <p className="text-red-400">Could not connect to model metadata service.</p>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-slate-800 text-cyan-400 rounded-lg text-sm font-semibold"
          >
            Retry
          </button>
        </div>
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

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto space-y-12">
        {/* Synthetic Banner */}
        {meta.synthetic && <Banner type="synthetic" />}

        {/* Hero Section */}
        <div className="space-y-4 border-b border-slate-800 pb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-xs font-medium">
            <Waves className="h-3.5 w-3.5" />
            <span>AI Oceanography</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-100">
            About OceanScope AI
          </h1>
          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            OceanScope AI reconstructs high-resolution, daily three-dimensional subsurface ocean temperature fields across the North Indian Ocean using surface satellite observations, climate teleconnection indices, and deep neural networks.
          </p>
        </div>

        {/* Dynamic Model Specifications Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-xl space-y-4">
          <h2 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-2">
            <Radio className="h-4 w-4" />
            <span>Active Deployment Parameters</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-850">
              <span className="text-slate-400 block text-[10px] font-sans">Run ID</span>
              <span className="font-semibold text-slate-100">{meta.run_id}</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-855">
              <span className="text-slate-400 block text-[10px] font-sans">Spatial Grid</span>
              <span className="font-semibold text-slate-100">
                {meta.grid_resolution_deg}° ({meta.region.lat_min}°–{meta.region.lat_max}°N, {meta.region.lon_min}°–{meta.region.lon_max}°E)
              </span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-850">
              <span className="text-slate-400 block text-[10px] font-sans">Depth Tiers</span>
              <span className="font-semibold text-slate-100">
                15 depths ({formatDepth(meta.depths_m[0])} to {formatDepth(meta.depths_m[meta.depths_m.length - 1])})
              </span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-850">
              <span className="text-slate-400 block text-[10px] font-sans">Available Dates</span>
              <span className="font-semibold text-cyan-300">
                {meta.available_dates.start} to {meta.available_dates.end}
              </span>
            </div>
          </div>
        </div>

        {/* The Scientific Problem */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2.5">
            <Layers className="h-5 w-5 text-cyan-400" />
            <span>The Challenge of Subsurface Ocean Observation</span>
          </h2>
          <div className="text-sm text-slate-300 space-y-3 leading-relaxed">
            <p>
              While Earth observation satellites measure surface characteristics (sea surface temperature, sea surface height, salinity, and surface winds) with high spatial and temporal coverage, satellites cannot penetrate into the ocean depths. Direct subsurface measurements rely almost exclusively on autonomous ARGO profiling floats, which drift freely and provide sparse point measurements separated by hundreds of kilometres and 10-day intervals.
            </p>
            <p>
              In the North Indian Ocean—encompassing the Arabian Sea and the Bay of Bengal—subsurface thermal stratification dictates cyclone intensification rates, monsoon progression, and marine heatwave development. Conventional numerical reanalysis models (such as GLORYS12) assimilate observations into coupled hydrodynamic equations, but require vast supercomputing infrastructure and can lag real-time observations by weeks.
            </p>
          </div>
        </div>

        {/* The Machine Learning Method */}
        <div className="space-y-4" id="method">
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2.5">
            <Cpu className="h-5 w-5 text-cyan-400" />
            <span>Our Methodology</span>
          </h2>
          <div className="text-sm text-slate-300 space-y-3 leading-relaxed">
            <p>
              OceanScope AI implements an end-to-end deep learning framework designed to reconstruct daily three-dimensional temperature fields down to 1500 metres:
            </p>
            <ul className="list-disc pl-5 space-y-2 text-slate-300">
              <li>
                <strong className="text-slate-100">Multi-Modal Surface Forcing:</strong> Inputs comprise sea level anomaly (SLA), sea surface temperature (SST), sea surface salinity (SSS), vector wind stress (CCMP), ocean surface currents (OSCAR), and surface heat flux anomalies (ERA5) stacked with 7-day and 60-day lag histories.
              </li>
              <li>
                <strong className="text-slate-100">Climate Teleconnection &amp; Float Conditioning:</strong> Basin-wide climate indices (Indian Ocean Dipole DMI, ENSO Niño 3.4) are broadcast across the grid as spatial scalar channels, alongside recent ARGO profile observations.
              </li>
              <li>
                <strong className="text-slate-100">U-Net Architecture:</strong> A small, highly regularized 2D U-Net processes multi-channel spatial fields to predict temperature anomalies relative to the World Ocean Atlas 2023 (WOA23) climatological mean across 15 depth tiers simultaneously.
              </li>
              <li>
                <strong className="text-slate-100">Staged Transfer Learning:</strong> The network is pretrained on GLORYS12 physical reanalysis and the GREP multi-reanalysis ensemble, and subsequently fine-tuned on verified ARGO in-situ CTD profiles.
              </li>
            </ul>
          </div>
        </div>

        {/* Live Dates Explained */}
        <div className="space-y-4 bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6">
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Calendar className="h-5 w-5 text-amber-400" />
            <span>Understanding &quot;Live&quot; Mode</span>
          </h2>
          <div className="text-sm text-slate-300 space-y-2 leading-relaxed">
            <p>
              Dates classified as <strong className="text-amber-300">Live</strong> represent near-real-time inferences generated approximately 3 days behind real time using operational satellite feeds before official numerical reanalyses are published.
            </p>
            <p className="text-xs text-slate-400">
              Because physical numerical reanalyses (such as GLORYS12) require several weeks of assimilation delay, no ground-truth reference or difference fields are available for live dates.
            </p>
          </div>
        </div>

        {/* Data Sources and Citations */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2.5">
            <Database className="h-5 w-5 text-cyan-400" />
            <span>Data Sources &amp; Acknowledgements</span>
          </h2>
          <p className="text-xs text-slate-400">
            We gratefully acknowledge the data providers whose open access initiatives make this work possible:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
            {dataSources.map((ds) => (
              <div
                key={ds.name}
                className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 text-xs space-y-1.5 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-200">{ds.name}</span>
                    <a
                      href={ds.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                    >
                      <span className="text-[10px]">Access</span>
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </div>
                  <p className="text-slate-400 text-[11px] mt-0.5">{ds.param}</p>
                </div>
                <div className="pt-2 border-t border-slate-850 text-[10px] text-slate-500 italic">
                  {ds.citation}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Scientific Limitations & Official Caveats */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2.5">
            <ShieldAlert className="h-5 w-5 text-amber-400" />
            <span>Official Scientific Caveats &amp; Limitations</span>
          </h2>
          <div className="space-y-3">
            {meta.caveats?.glorys && (
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs space-y-1">
                <span className="font-semibold text-cyan-300 uppercase tracking-wider text-[10px] block">
                  GLORYS Reanalysis Reference
                </span>
                <p className="text-slate-300 leading-relaxed">{meta.caveats.glorys}</p>
              </div>
            )}
            {meta.caveats?.argo && (
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs space-y-1">
                <span className="font-semibold text-amber-300 uppercase tracking-wider text-[10px] block">
                  ARGO Float Independence
                </span>
                <p className="text-slate-300 leading-relaxed">{meta.caveats.argo}</p>
              </div>
            )}
            {meta.caveats?.live && (
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs space-y-1">
                <span className="font-semibold text-amber-300 uppercase tracking-wider text-[10px] block">
                  Operational Near-Real-Time Mode
                </span>
                <p className="text-slate-300 leading-relaxed">{meta.caveats.live}</p>
              </div>
            )}
            {meta.caveats?.synthetic && (
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs space-y-1">
                <span className="font-semibold text-rose-300 uppercase tracking-wider text-[10px] block">
                  Demo Environment Notice
                </span>
                <p className="text-slate-300 leading-relaxed">{meta.caveats.synthetic}</p>
              </div>
            )}
          </div>
        </div>

        {/* Team and Navigation */}
        <div className="border-t border-slate-800 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <div>OceanScope AI • Created by Aryan Chettri &amp; AI Pair Programming Team</div>
          <div className="flex items-center gap-4">
            <Link href="/explore" className="text-cyan-400 hover:underline">
              ← Explorer Map
            </Link>
            <Link href="/validation" className="text-cyan-400 hover:underline">
              Validation Benchmarks →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
