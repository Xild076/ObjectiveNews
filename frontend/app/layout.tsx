import type { Metadata } from "next";
import { Space_Grotesk, Newsreader } from "next/font/google";
import "./globals.css";

const grotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-grotesk" });
const newsreader = Newsreader({ subsets: ["latin"], variable: "--font-news" });

export const metadata: Metadata = {
  title: "Objective News Studio",
  description: "Cluster, summarize, and compare news narratives with Objective News",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${grotesk.variable} ${newsreader.variable}`}>
      <body>{children}</body>
    </html>
  );
}
