"use client";

import { useState } from "react";
import { Chart } from "recharts";

export default function Page({ show }: { show: boolean }) {
  if (show) {
    const [n, setN] = useState(0);
    setN(n + 1);
  }
  const total: number = "forty-two";
  return <main>{total}<Chart /></main>;
}
