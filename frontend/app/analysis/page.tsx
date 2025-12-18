"use client";

import { useState } from "react";
import Link from "next/link";

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export default function AnalysisPage() {
  const [text, setText] = useState("");
  const [linkCount, setLinkCount] = useState(10);
  const [diverseLinks, setDiverseLinks] = useState(true);
  const [summarizeLevel, setSummarizeLevel] = useState("medium");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter a topic or URL");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          link_n: linkCount,
          diverse_links: diverseLinks,
          summarize_level: summarizeLevel,
        }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResults(data);
    } catch (err) {
      setError((err as Error).message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setText("");
    setError(null);
  };

  return (
    <main>
      <section className="section">
        <Link href="/" style={{ color: "var(--accent)", marginBottom: "1rem", display: "inline-block" }}>
          ← Back to Home
        </Link>
        <h1>📊 Article Analysis</h1>
        <p style={{ color: "var(--muted)" }}>
          Enter a topic or URL to retrieve multiple sources, identify core narratives, and see them as summarized, objective
          clusters with reliability scores.
        </p>
      </section>

      {results ? (
        <section className="section">
          <button onClick={handleReset} style={{ marginBottom: "1.5rem" }}>
            ← New Analysis
          </button>
          <h2>Analysis Results</h2>
          <p style={{ color: "var(--muted)" }}>Found {results.clusters?.length || 0} narratives</p>

          {results.clusters && results.clusters.length > 0 ? (
            <div className="cluster-results">
              {results.clusters.map((cluster: any, idx: number) => (
                <div key={idx} className="cluster-card">
                  <div className="badge">Narrative #{idx + 1}</div>
                  {cluster.representative && (
                    <>
                      <h4>{cluster.representative.text}</h4>
                      <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
                        {cluster.representative.source || "Unknown source"}
                      </p>
                    </>
                  )}
                  {cluster.summary && (
                    <p style={{ marginTop: "1rem", lineHeight: 1.6 }}>{cluster.summary}</p>
                  )}
                  {cluster.reliability !== undefined && (
                    <div style={{ marginTop: "1rem" }}>
                      <span className="badge">Reliability: {cluster.reliability.toFixed(1)}%</span>
                    </div>
                  )}
                  {cluster.sentences && cluster.sentences.length > 0 && (
                    <details style={{ marginTop: "1rem" }}>
                      <summary style={{ cursor: "pointer", color: "var(--accent)" }}>
                        View {cluster.sentences.length} contributing sentences
                      </summary>
                      <ul style={{ marginTop: "0.5rem", paddingLeft: "1.25rem" }}>
                        {cluster.sentences.slice(0, 5).map((s: any, i: number) => (
                          <li key={i} style={{ marginBottom: "0.5rem", color: "var(--muted)" }}>
                            {s.text}
                            {s.source && <span style={{ color: "var(--accent)" }}> ({s.source})</span>}
                          </li>
                        ))}
                        {cluster.sentences.length > 5 && <li>+{cluster.sentences.length - 5} more</li>}
                      </ul>
                    </details>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: "var(--muted)" }}>No narratives found. Try a different topic.</p>
          )}
        </section>
      ) : (
        <section className="section">
          <div className="card">
            <h3>Configure Analysis</h3>
            <label style={{ display: "block", marginTop: "1rem", marginBottom: "0.5rem" }}>
              Topic or URL
            </label>
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g., 'global economic outlook' or paste a news URL"
            />

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginTop: "1.5rem" }}>
              <div>
                <label style={{ display: "block", marginBottom: "0.5rem" }}>Articles to Fetch</label>
                <input
                  type="number"
                  value={linkCount}
                  onChange={(e) => setLinkCount(Number(e.target.value))}
                  min={5}
                  max={20}
                />
              </div>
              <div>
                <label style={{ display: "block", marginBottom: "0.5rem" }}>Summarization</label>
                <select value={summarizeLevel} onChange={(e) => setSummarizeLevel(e.target.value)}>
                  <option value="fast">Fast</option>
                  <option value="medium">Medium</option>
                  <option value="best">Best</option>
                </select>
              </div>
            </div>

            <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "1rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={diverseLinks}
                onChange={(e) => setDiverseLinks(e.target.checked)}
              />
              Use diverse sources
            </label>

            <button onClick={handleAnalyze} disabled={loading} style={{ marginTop: "1.5rem" }}>
              {loading ? "Analyzing... (this may take 30-60s)" : "Analyze Topic"}
            </button>

            {error && <p style={{ color: "#ff6f61", marginTop: "1rem" }}>{error}</p>}

            <div style={{ marginTop: "2rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
              <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginBottom: "0.75rem" }}>Try an example:</p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button
                  onClick={() => setText("US elections")}
                  style={{ background: "rgba(255,255,255,0.05)", padding: "0.5rem 1rem", fontSize: "0.9rem" }}
                >
                  US elections
                </button>
                <button
                  onClick={() => setText("AI regulation")}
                  style={{ background: "rgba(255,255,255,0.05)", padding: "0.5rem 1rem", fontSize: "0.9rem" }}
                >
                  AI regulation
                </button>
                <button
                  onClick={() => setText("Climate change")}
                  style={{ background: "rgba(255,255,255,0.05)", padding: "0.5rem 1rem", fontSize: "0.9rem" }}
                >
                  Climate change
                </button>
              </div>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
