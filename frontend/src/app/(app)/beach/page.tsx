"use client";

import React from "react";
import dynamic from "next/dynamic";
import Link from "next/link";

import { useMeta } from "../../../lib/api/hooks";
import { Loading } from "../../../components/common/Loading";

// Beach mode code loads only on this route (PRD_FRONTEND §4 Meta features)
const BeachMode = dynamic(() => import("../../../components/beach/BeachMode").then((m) => m.BeachMode), {
  ssr: false,
  loading: () => <Loading label="Loading beach mode..." />,
});

export default function BeachPage() {
  const { data: meta, isLoading } = useMeta();
  if (isLoading) return <Loading fullScreen label="Loading beach mode..." />;
  if (!meta?.features?.beach) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 p-6 text-center text-slate-700 dark:text-slate-300">
        <p>Beach mode isn&apos;t available right now.</p>
        <Link href="/explore" className="underline text-cyan-700 dark:text-cyan-400">
          Open the map
        </Link>
      </div>
    );
  }
  return <BeachMode meta={meta} />;
}
