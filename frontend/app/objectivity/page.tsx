"use client";

import { useState } from "react";
import Link from "next/link";

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

function renderDiff(original: string, objectified: string) {
  const origWords = original.split(/\s+/);
  const objWords = objectified.split(/\s+/);
  
  // Simple word-level diff visualization
  const result: JSX.Element[] = [];
  let i = 0, j = 0;
  
  while (i < origWords.length || j < objWords.length) {
    if (i < origWords.length && j < objWords.length && origWords[i] === objWords[j]) {
      result.push(<span key={`${i}-${j}`}>{origWords[i]} </span>);
      i++;
      j++;
    } else if (i < origWords.length) {
      result.push(
        <span key={`del-${i}`} style={{ backgroundColor: "#ffebe9", color: "#d9534f", textDecoration: "line-through", padding: "2px 4px", borderRadius: "4px" }}>
          {origWords[i]}
        </span>
      );
      result.push(<span key={`space-${i}`}> </span>);
      i++;
    } else {
      result.push(
        <span key={`add-${j}`} style={{ backgroundColor: "#dbeafe", color: "#2563eb", fontWeight: "bold", padding: "2px 4px", borderRadius: "4px" }}>
          {objWords[j]}
        </span>
      );
      result.push(<span key={`space-${j}`}> </span>);
      j++;
    }
  }
  
  return result;
}

export default function ObjectivityPage() {
  const defaultText = "The corrupt politician delivered a disastrous speech, infuriating the brave citizens who are fighting for justice.";
  const [text, setText] = useState(defaultText);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleObjectify = async () => {
    if (!text.trim()) {
      setError("Please enter a sentence");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/objectify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError((err as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <section className="section">
        <Link href="/" style={{ color: "var(--accent)", marginBottom: "1rem", display: "inline-block" }}>
          ← Back to Home
        </Link>
        <h1>⚖️ Objectivity Playground</h1>
        <p style={{ color: "var(--muted)" }}>
          Test the objectification engine. Enter a subjective sentence and see how the system rewrites it to be more neutral.
        </p>
        <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(255,255,255,0.03)", borderRadius: "12px" }}>
          <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.95rem", lineHeight: 1.6 }}>
            <strong>How to read the results:</strong><br />
            • Objectivity Score: 0.0 to 1.0, where 1.0 is completely neutral<br />
            • <span style={{ color: "#d9534f", textDecoration: "line-through" }}>Red text</span> was removed for being too subjective<br />
            • <span style={{ backgroundColor: "#dbeafe", color: "#2563eb", fontWeight: "bold", padding: "2px 4px", borderRadius: "4px" }}>Blue text</span> was added or modified to be more neutral
          </p>
        </div>
      </section>

      <section className="section">
        <div className="card">
          <h3>Enter a sentence to objectify</h3>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            placeholder="Paste a subjective sentence from any source..."
            style={{ marginTop: "1rem" }}
          />

          <button onClick={handleObjectify} disabled={loading} style={{ marginTop: "1rem" }}>
            {loading ? "Analyzing and rewriting..." : "Objectify Sentence"}
          </button>

          {error && <p style={{ color: "#ff6f61", marginTop: "1rem" }}>{error}</p>}
        </div>

        {result && (
          <div style={{ marginTop: "2rem" }}>
            <h3>Results</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
              <div className="card">
                <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.9rem" }}>Original Score</p>
                <p style={{ margin: "0.5rem 0 0", fontSize: "2rem", fontWeight: "bold" }}>{result.original_score.toFixed(3)}</p>
              </div>
              <div className="card">
                <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.9rem" }}>New Score</p>
                <p style={{ margin: "0.5rem 0 0", fontSize: "2rem", fontWeight: "bold", color: "var(--accent)" }}>
                  {result.new_score.toFixed(3)}
                </p>
                <p style={{ margin: "0.25rem 0 0", color: result.improvement >= 0 ? "#4ade80" : "#ff6f61", fontSize: "0.9rem" }}>
                  {result.improvement >= 0 ? "+" : ""}{result.improvement.toFixed(3)}
                </p>
              </div>
            </div>

            <div className="card">
              <h4 style={{ marginTop: 0 }}>Detailed Transformation View</h4>
              <div style={{ padding: "1rem", background: "rgba(0,0,0,0.3)", borderRadius: "12px", lineHeight: 1.7 }}>
                {renderDiff(result.original_text, result.objectified_text)}
              </div>
            </div>

            <div className="card" style={{ marginTop: "1rem" }}>
              <h4 style={{ marginTop: 0 }}>Final Neutral Text</h4>
              <p style={{ margin: 0, lineHeight: 1.7 }}>{result.objectified_text}</p>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
