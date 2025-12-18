"use client";

import { useMemo, useState } from "react";

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

const initialSentence = "Political leaders met on Tuesday to discuss a potential ceasefire.";
const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "/api";
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

const toSentencePayload = (articles: typeof initialArticles) => {
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
};

export default function Home() {
  const [sentences, setSentences] = useState<string[]>([initialSentence]);
  const [labels, setLabels] = useState<number[]>([]);
  const [sentLoading, setSentLoading] = useState(false);
  const [sentError, setSentError] = useState<string | null>(null);

  const [articles, setArticles] = useState(initialArticles);
  const [articleClusters, setArticleClusters] = useState<ClusterDict[]>([]);
  const [articleLoading, setArticleLoading] = useState(false);
  const [articleError, setArticleError] = useState<string | null>(null);

  const articleSentenceCount = useMemo(() => toSentencePayload(articles).length, [articles]);

  const handleSentenceChange = (index: number, value: string) => {
    setSentences((prev) => prev.map((sentence, i) => (i === index ? value : sentence)));
  };

  const addSentence = () => setSentences((prev) => [...prev, ""]);

  const handleClusterSentences = async () => {
    const payload = sentences.map((s) => s.trim()).filter(Boolean);
    if (!payload.length) {
      setSentError("Add at least one sentence to cluster.");
      return;
    }
    setSentError(null);
    setSentLoading(true);
    try {
      const res = await fetch(`${apiBase}/cluster`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences: payload }),
      });
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      const data = await res.json();
      setLabels(data.labels ?? []);
    } catch (error) {
      setSentError((error as Error).message);
      setLabels([]);
    } finally {
      setSentLoading(false);
    }
  };

  const handleArticleChange = (index: number, field: "text" | "source" | "author", value: string) => {
    setArticles((prev) => prev.map((article, i) => (i === index ? { ...article, [field]: value } : article)));
  };

  const addArticle = () => {
    setArticles((prev) => [...prev, { source: "New Outlet", author: "", text: "" }]);
  };

  const handleClusterArticles = async () => {
    const payload = toSentencePayload(articles);
    if (!payload.length) {
      setArticleError("Provide at least one full sentence across your articles.");
      return;
    }
    setArticleError(null);
    setArticleLoading(true);
    try {
      const res = await fetch(`${apiBase}/cluster-texts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences: payload }),
      });
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      const data = await res.json();
      setArticleClusters(data.clusters ?? []);
    } catch (error) {
      setArticleError((error as Error).message);
      setArticleClusters([]);
    } finally {
      setArticleLoading(false);
    }
  };

  return (
    <main>
      <section className="section hero">
        <p className="badge">Objective News Studio · beta</p>
        <h1>Trace every narrative, from rumor to reality.</h1>
        <p>
          Compare reporting angles, surface consensus facts, and flag contradictions using the same clustering engine that powers the
          research pipeline.
        </p>
        <div className="pills">
          <span className="pill">Sentence embeddings + attention</span>
          <span className="pill">Contextual clustering</span>
          <span className="pill">Representative pull quotes</span>
        </div>
      </section>

      <section className="section">
        <div className="grid">
          <div className="card">
            <div className="badge">Quick test</div>
            <h3>Cluster loose sentences</h3>
            <p>Paste any statements you want to compare. The API returns a cluster ID per sentence.</p>
            {sentences.map((sentence, index) => (
              <textarea
                key={index}
                value={sentence}
                onChange={(event) => handleSentenceChange(index, event.target.value)}
                placeholder="Sentence..."
                rows={3}
                style={{ marginTop: index === 0 ? "1rem" : "0.75rem" }}
              />
            ))}
            <button type="button" onClick={addSentence} style={{ background: "rgba(255,255,255,0.1)", color: "var(--text)", border: "1px solid rgba(255,255,255,0.2)" }}>
              + Add sentence
            </button>
            <button type="button" onClick={handleClusterSentences} disabled={sentLoading}>
              {sentLoading ? "Clustering…" : "Cluster sentences"}
            </button>
            {sentError && <p style={{ color: "#ff6f61", margin: 0 }}>{sentError}</p>}
            {Boolean(labels.length) && (
              <div>
                <p style={{ color: "var(--muted)" }}>Labels returned:</p>
                <div className="pills" style={{ justifyContent: "flex-start" }}>
                  {labels.map((label, index) => (
                    <span key={`${label}-${index}`} className="pill" style={{ color: "var(--text)" }}>
                      #{label}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="badge">Deep dive</div>
            <h3>Compare full articles</h3>
            <p>Drop two or more write-ups. We split them into sentences, cluster them, and highlight the center of each narrative.</p>
            {articles.map((article, index) => (
              <div key={index} style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.5rem" }}>
                <input
                  type="text"
                  value={article.source}
                  placeholder="Source"
                  onChange={(event) => handleArticleChange(index, "source", event.target.value)}
                />
                <input
                  type="text"
                  value={article.author}
                  placeholder="Reporter"
                  onChange={(event) => handleArticleChange(index, "author", event.target.value)}
                />
                <textarea
                  value={article.text}
                  placeholder="Paste the article paragraphs here..."
                  rows={4}
                  onChange={(event) => handleArticleChange(index, "text", event.target.value)}
                />
              </div>
            ))}
            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <button type="button" onClick={addArticle} style={{ background: "transparent", color: "var(--text)", border: "1px solid rgba(255,255,255,0.2)" }}>
                + Add article
              </button>
              <button type="button" onClick={handleClusterArticles} disabled={articleLoading}>
                {articleLoading ? "Clustering…" : `Cluster ${articleSentenceCount} sentences`}
              </button>
            </div>
            {articleError && <p style={{ color: "#ff6f61", margin: 0 }}>{articleError}</p>}
          </div>
        </div>

        {Boolean(articleClusters.length) && (
          <div style={{ marginTop: "3rem" }}>
            <div className="badge">Narrative map</div>
            <h3 style={{ marginTop: "0.5rem" }}>Clustered articles</h3>
            <div className="cluster-results">
              {articleClusters.map((cluster) => (
                <div key={cluster.label} className="cluster-card">
                  <div className="badge">Cluster #{cluster.label}</div>
                  {cluster.representative && (
                    <>
                      <h4>{cluster.representative.text}</h4>
                      <p style={{ color: "var(--muted)", marginTop: 0 }}>
                        {cluster.representative.source ?? "Unknown outlet"}
                        {cluster.representative.author ? ` • ${cluster.representative.author}` : ""}
                      </p>
                    </>
                  )}
                  <ul>
                    {cluster.sentences.slice(0, 5).map((sentence, index) => (
                      <li key={`${cluster.label}-${index}`}>
                        {sentence.text}
                        {sentence.source ? <span style={{ color: "var(--accent)", marginLeft: "0.25rem" }}>({sentence.source})</span> : null}
                      </li>
                    ))}
                    {cluster.sentences.length > 5 && <li>+{cluster.sentences.length - 5} more</li>}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
