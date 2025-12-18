"use client";

import Link from "next/link";

export default function Home() {
  return (
    <main>
      <section className="section hero">
        <h1>Objective News Studio</h1>
        <p>
          Analyze topics across sources, cluster narratives, and see objective summaries with reliability scores.
          Built to combat misinformation through semantic analysis and source verification.
        </p>
        <div className="pills">
          <span className="pill">Multi-source analysis</span>
          <span className="pill">Narrative clustering</span>
          <span className="pill">Objectivity scoring</span>
          <span className="pill">Reliability metrics</span>
        </div>
      </section>

      <section className="section">
        <h2 style={{ marginTop: 0 }}>Features</h2>
        <div className="grid">
          <Link href="/analysis" className="card feature-card">
            <div className="badge">Core Feature</div>
            <h3>📊 Article Analysis</h3>
            <p>
              Enter a topic or URL to fetch multiple sources, identify core narratives, and see summarized clusters
              with reliability scores.
            </p>
            <span className="link-arrow">Analyze Articles →</span>
          </Link>

          <Link href="/objectivity" className="card feature-card">
            <div className="badge">AI Rewriting</div>
            <h3>⚖️ Objectivity Playground</h3>
            <p>
              Test the objectification engine. Enter a subjective sentence and see how the system rewrites it to be more neutral.
            </p>
            <span className="link-arrow">Objectify Text →</span>
          </Link>

          <Link href="/clustering" className="card feature-card">
            <div className="badge">Advanced</div>
            <h3>🔗 Sentence Clustering</h3>
            <p>
              Compare reporting angles, surface consensus facts, and cluster similar sentences using embeddings and attention models.
            </p>
            <span className="link-arrow">Cluster Sentences →</span>
          </Link>

          <Link href="/utilities" className="card feature-card">
            <div className="badge">Labs</div>
            <h3>🧰 Utilities Explorer</h3>
            <p>
              Check domain reliability and rank synonyms by objectivity—the same utilities from the Streamlit app.
            </p>
            <span className="link-arrow">Open Utilities →</span>
          </Link>
        </div>
      </section>

      <section className="section" style={{ marginTop: "2rem" }}>
        <h3>The Status Quo of Misinformation</h3>
        <p style={{ color: "var(--muted)", lineHeight: 1.7 }}>
          Misinformation is rising with new technologies and polarization. Echo chambers, algorithmic feeds, and
          content velocity make it harder to separate fact from fiction. Surveys report that the majority of people see
          misinformation regularly—often more than they realize.
        </p>
        <p style={{ color: "var(--muted)", lineHeight: 1.7, marginTop: "1rem" }}>
          Objective News tackles this by fetching diverse sources, clustering narratives, neutralizing biased language,
          and assigning transparent reliability scores.
        </p>
        <div className="pills" style={{ justifyContent: "flex-start", marginTop: "1rem" }}>
          <span className="pill">Fetch diverse articles</span>
          <span className="pill">Cluster core narratives</span>
          <span className="pill">Objectify language</span>
          <span className="pill">Score reliability</span>
        </div>
      </section>
    </main>
  );
}
