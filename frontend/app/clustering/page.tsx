"use client";

import { useMemo, useState } from "react";
import Link from "next/link";

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

type Sentence = {
  text: string;
  source?: string | null;
  author?: string | null;
  date?: string | null;
};

type ClusterDict = {
  label: number;
  sentences: Sentence[];
  representative?: Sentence;
  representative_with_context?: [Sentence | null, Sentence | null];
};

const initialArticles = [
  {
    source: "Wire Service",
    author: "J. Patel",
    text: "Government envoys met overnight to renegotiate the ceasefire proposal. Early drafts suggest humanitarian corridors would reopen within 72 hours.",
  },
  {
    source: "Regional Desk",
    author: "M. Lopez",
    text: "Local observers reported sporadic shelling despite optimism around talks. Aid groups warned that fuel shortages remain critical.",
  },
];

function toSentencePayload(articles: typeof initialArticles) {
  const sentences: Sentence[] = [];
  articles.forEach(({ text, source, author }) => {
    text
      .split(/\n+|(?<=[.!?])\s+/)
      .map((chunk) => chunk.trim())
      .filter(Boolean)
      .forEach((chunk) => {
        sentences.push({ text: chunk, source, author });
      });
  });
  return sentences;
}

export default function ClusteringPage() {
  const [articles, setArticles] = useState(initialArticles);
  const [clusters, setClusters] = useState<ClusterDict[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sentenceCount = useMemo(() => toSentencePayload(articles).length, [articles]);

  const handleArticleChange = (index: number, field: "text" | "source" | "author", value: string) => {
    setArticles((prev) => prev.map((article, i) => (i === index ? { ...article, [field]: value } : article)));
  };

  const addArticle = () => {
    setArticles((prev) => [...prev, { source: "New Outlet", author: "", text: "" }]);
  };

  const handleCluster = async () => {
    const payload = toSentencePayload(articles);
    if (!payload.length) {
      setError("Provide at least one full sentence across your articles.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/cluster-texts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences: payload }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data = await res.json();
      setClusters(data.clusters ?? []);
    } catch (err) {
      setError((err as Error).message);
      setClusters([]);
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
        <h1>🔗 Sentence Clustering</h1>
        <p style={{ color: "var(--muted)" }}>
          Compare reporting angles, surface consensus facts, and cluster similar sentences using embeddings and attention models.
        </p>
      </section>

      <section className="section">
        <div className="card">
          <h3>Add Articles to Cluster</h3>
          <p style={{ color: "var(--muted)" }}>
            Drop two or more write-ups. We split them into sentences, cluster them, and highlight the center of each narrative.
          </p>

          {articles.map((article, index) => (
            <div key={index} style={{ marginTop: index === 0 ? "1.5rem" : "1rem", padding: "1rem", background: "rgba(255,255,255,0.03)", borderRadius: "12px" }}>
              <div style={{ display: "flex", gap: "0.75rem", marginBottom: "0.75rem" }}>
                <input
                  type="text"
                  value={article.source}
                  placeholder="Source"
                  onChange={(e) => handleArticleChange(index, "source", e.target.value)}
                  style={{ flex: 1 }}
                />
                <input
                  type="text"
                  value={article.author}
                  placeholder="Reporter"
                  onChange={(e) => handleArticleChange(index, "author", e.target.value)}
                  style={{ flex: 1 }}
                />
              </div>
              <textarea
                value={article.text}
                placeholder="Paste the article paragraphs here..."
                rows={4}
                onChange={(e) => handleArticleChange(index, "text", e.target.value)}
              />
            </div>
          ))}

          <div style={{ display: "flex", gap: "0.75rem", marginTop: "1.5rem" }}>
            <button
              type="button"
              onClick={addArticle}
              style={{ background: "rgba(255,255,255,0.05)", color: "var(--text)", border: "1px solid rgba(255,255,255,0.2)", flex: 1 }}
            >
              + Add Article
            </button>
            <button type="button" onClick={handleCluster} disabled={loading} style={{ flex: 2 }}>
              {loading ? "Clustering..." : `Cluster ${sentenceCount} Sentences`}
            </button>
          </div>

          {error && <p style={{ color: "#ff6f61", marginTop: "1rem" }}>{error}</p>}
        </div>

        {clusters.length > 0 && (
          <div style={{ marginTop: "2rem" }}>
            <h3>Narrative Map</h3>
            <p style={{ color: "var(--muted)" }}>Found {clusters.length} distinct narrative clusters</p>

            <div className="cluster-results">
              {clusters.map((cluster) => (
                <div key={cluster.label} className="cluster-card">
                  <div className="badge">Cluster #{cluster.label}</div>
                  {cluster.representative && (
                    <>
                      <h4>{cluster.representative.text}</h4>
                      <p style={{ color: "var(--muted)", marginTop: "0.5rem", fontSize: "0.9rem" }}>
                        {cluster.representative.source ?? "Unknown outlet"}
                        {cluster.representative.author ? ` • ${cluster.representative.author}` : ""}
                      </p>
                    </>
                  )}
                  <details style={{ marginTop: "1rem" }}>
                    <summary style={{ cursor: "pointer", color: "var(--accent)" }}>
                      View {cluster.sentences.length} contributing sentences
                    </summary>
                    <ul style={{ marginTop: "0.75rem", paddingLeft: "1.25rem" }}>
                      {cluster.sentences.map((sentence, index) => (
                        <li key={`${cluster.label}-${index}`} style={{ marginBottom: "0.5rem", color: "var(--muted)" }}>
                          {sentence.text}
                          {sentence.source && (
                            <span style={{ color: "var(--accent)", marginLeft: "0.25rem" }}>({sentence.source})</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </details>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
