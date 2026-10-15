"use client";

import { useState } from "react";

type Tier = { name: string; monthly: number; features: string[] };

const TIERS: Tier[] = [
  { name: "Starter", monthly: 9, features: ["1 project", "Email support"] },
  { name: "Pro", monthly: 29, features: ["10 projects", "Priority support"] },
  { name: "Team", monthly: 79, features: ["Unlimited projects", "SSO"] },
];

export default function PricingTable() {
  const [yearly, setYearly] = useState(false);
  const price = (t: Tier) => (yearly ? Math.round(t.monthly * 12 * 0.8) : t.monthly);

  return (
    <section>
      <button
        type="button"
        aria-pressed={yearly}
        onClick={() => setYearly((y) => !y)}
        className="my-4 rounded border px-3 py-1"
      >
        {yearly ? "Billed yearly (save 20%)" : "Billed monthly"}
      </button>
      <div className="grid gap-4 md:grid-cols-3">
        {TIERS.map((t) => (
          <article key={t.name} className="rounded-lg border p-6">
            <h2 className="text-xl font-semibold">{t.name}</h2>
            <p className="text-2xl">
              ${price(t)}
              <span className="text-sm">/{yearly ? "yr" : "mo"}</span>
            </p>
            <ul>
              {t.features.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}
