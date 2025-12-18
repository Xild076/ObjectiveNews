"use client";

import { useState } from "react";
import Link from "next/link";

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export default function UtilitiesPage() {
  const [word, setWord] = useState("beautiful");
  const [synonyms, setSynonyms] = useState<any[]>([]);
  const [synLoading, setSynLoading] = useState(false);
  const [synError, setSynError] = useState<string | null>(null);

  const [domain, setDomain] = useState("nytimes.com");
  const [domainResult, setDomainResult] = useState<any | null>(null);
  const [domLoading, setDomLoading] = useState(false);
  const [domError, setDomError] = useState<string | null>(null);

  const fetchSynonyms = async () => {
    if (!word.trim()) {
      setSynError("Enter a word");
      return;
    }
    setSynError(null);
    setSynLoading(true);
    try {
      const res = await fetch(`${apiBase}/synonyms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word, topn: 15 })
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setSynonyms(data.synonyms ?? []);
    } catch (err) {
      setSynError((err as Error).message);
      setSynonyms([]);
    } finally {
      setSynLoading(false);
    }
  };

  const fetchDomain = async () => {
    if (!domain.trim()) {
      setDomError("Enter a domain");
      return;
    }
    setDomError(null);
    setDomLoading(true);
    try {
      const res = await fetch(`${apiBase}/domain-reliability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain })
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setDomainResult(data);
    } catch (err) {
      setDomError((err as Error).message);
      setDomainResult(null);
    } finally {
      setDomLoading(false);
    }
  };

  return (
    <main>
      <section className="section">
        <Link href="/" style={{ color: "var(--accent)", marginBottom: "1rem", display: "inline-block" }}>
          ← Back to Home
        </Link>
        <h1>🧰 Utilities Explorer</h1>
        <p style={{ color: "var(--muted)", marginBottom: "1.5rem" }}>
          Explore the underlying components of the analysis engine: synonym objectivity ranking and domain reliability checking.
        </p>
      </section>

      <section className="section" style={{ display: "grid", gap: "1.5rem" }}>
        <div className="card">
          <div className="badge">Synonym Objectivity Ranker</div>
          <p style={{ color: "var(--muted)" }}>
            Find synonyms for a word, ranked by objectivity (1.0 = most neutral). Try adjectives or loaded terms.
          </p>
          <input
            type="text"
            value={word}
            onChange={(e) => setWord(e.target.value)}
            placeholder="Enter a word"
            style={{ marginTop: "0.5rem" }}
          />
          <button onClick={fetchSynonyms} disabled={synLoading} style={{ marginTop: "0.75rem" }}>
            {synLoading ? "Fetching..." : "Rank synonyms"}
          </button>
          {synError && <p style={{ color: "#ff6f61", marginTop: "0.5rem" }}>{synError}</p>}

          {synonyms.length > 0 && (
            <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>
              {synonyms.map((syn, idx) => (
                <div key={idx} className="card" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <strong>{idx + 1}. {syn.word}</strong>
                      {syn.definition && <p style={{ margin: 0, color: "var(--muted)" }}>{syn.definition}</p>}
                    </div>
                    <div className="badge">{syn.objectivity?.toFixed(3)}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="badge">Domain Reliability Checker</div>
          <p style={{ color: "var(--muted)" }}>
            Check the stored reliability score for a news domain. Score ranges from -1 (low) to +1 (high).
          </p>
          <input
            type="text"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            placeholder="e.g., cnn.com"
            style={{ marginTop: "0.5rem" }}
          />
          <button onClick={fetchDomain} disabled={domLoading} style={{ marginTop: "0.75rem" }}>
            {domLoading ? "Checking..." : "Check reliability"}
          </button>
          {domError && <p style={{ color: "#ff6f61", marginTop: "0.5rem" }}>{domError}</p>}

          {domainResult && (
            <div style={{ marginTop: "1rem" }}>
              <h3 style={{ marginTop: 0 }}>Reliability for {domainResult.domain}</h3>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
                <div className="card" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)" }}>
                  <p style={{ margin: 0, color: "var(--muted)" }}>Score [-1 to 1]</p>
                  <p style={{ margin: "0.35rem 0 0", fontSize: "1.4rem", fontWeight: 700 }}>{domainResult.score?.toFixed(3)}</p>
                </div>
                <div className="card" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)" }}>
                  <p style={{ margin: 0, color: "var(--muted)" }}>Percentage</p>
                  <p style={{ margin: "0.35rem 0 0", fontSize: "1.4rem", fontWeight: 700 }}>{domainResult.percent?.toFixed(1)}%</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
