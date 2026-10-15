import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CodeOrch sandbox",
  description: "Default layout. Generated code may replace this file.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
