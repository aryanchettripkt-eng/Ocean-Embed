import React from "react";
import { Header } from "../../components/common/Header";

export default function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <Header />
      <div className="flex-1 flex flex-col">{children}</div>
    </>
  );
}
