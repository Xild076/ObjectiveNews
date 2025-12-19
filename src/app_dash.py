import csv
import json
import io
import logging
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List
from urllib.parse import parse_qs

import dash
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

GLOBAL_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1021;
    --card: rgba(255, 255, 255, 0.9);
    --accent: #3b82f6;
    --accent-2: #22c55e;
    --muted: #6b7280;
    --border: rgba(255, 255, 255, 0.4);
    --ink: #0f172a;
    --shell: #f8fafc;
}

body {
    background: radial-gradient(circle at 20% 20%, rgba(59,130,246,0.12), transparent 25%),
                radial-gradient(circle at 80% 0%, rgba(16,185,129,0.12), transparent 25%),
                var(--bg);
    color: var(--ink);
    font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
}

.page-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 8px 12px 24px;
}

.hero-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(16,185,129,0.12));
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 18px 60px rgba(0,0,0,0.2);
}

.glass-card {
    background: var(--card);
    border-radius: 18px;
    border: 1px solid var(--border);
    box-shadow: 0 12px 40px rgba(0,0,0,0.08);
    backdrop-filter: blur(10px);
}

.pill {
    background: #e5edff;
    color: #1e3a8a;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}

.metric-card h3 {
    margin: 0;
}

.section {
    padding-top: 16px;
    padding-bottom: 16px;
}

.chip {
    display: inline-block;
    background: #eef2ff;
    color: #312e81;
    border-radius: 999px;
    padding: 4px 10px;
    margin-right: 6px;
    margin-bottom: 6px;
    font-size: 12px;
}

.divider {
    height: 1px;
    width: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.12), transparent);
    margin: 14px 0;
}

.dash-tabs .tab { background: transparent !important; color: #1f2937; font-weight: 600; border: none; }
.dash-tabs .tab--selected { background: #e0e7ff !important; color: #111827; border-bottom: 2px solid #3b82f6 !important; }
.glass-card .card-body { padding: 18px 20px; }
.form-label { font-weight: 600; color: #111827; }
"""

from article_analysis import article_analysis
from objectify.objectify import calculate_objectivity, objectify_text
from objectify.synonym import get_synonyms
from reliability import _load_source_df, get_source_label, normalize_domain
from utility import ensure_nltk_data, load_keybert

# Ensure required data/models are available before serving requests
ensure_nltk_data()
logger = logging.getLogger(__name__)

KW_MODEL = load_keybert()


def generate_cluster_title(text: str) -> str:
    if not text:
        return "Untitled Narrative"
    try:
        kws = KW_MODEL.extract_keywords(
            text,
            keyphrase_ngram_range=(2, 4),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=1,
        )
        if kws:
            return kws[0][0].title()
    except Exception:
        pass
    return (text[:60].strip().title() + "...") if len(text) > 60 else text.title()


def _serialize_sentence(s: Any) -> Dict[str, Any]:
    s0 = s[0] if isinstance(s, tuple) else s
    return {
        "text": getattr(s0, "text", ""),
        "source": getattr(s0, "source", "unknown") or "unknown",
        "date": getattr(s0, "date", None),
    }


def serialize_cluster(c: Dict[str, Any]) -> Dict[str, Any]:
    rep = c.get("representative")
    rep_text = getattr(rep, "text", "") if rep else ""
    return {
        "summary": c.get("summary", ""),
        "reliability": float(c.get("reliability", 0.0) or 0.0),
        "representative": rep_text,
        "sentences": [_serialize_sentence(s) for s in c.get("sentences", [])],
    }


@lru_cache(maxsize=256)
def fetch_source_reliability(domain: str) -> float:
    if not domain:
        return 0.0
    try:
        normalized = normalize_domain(domain)
        df = _load_source_df()
        score, _ = get_source_label(normalized, df)
        if score is None:
            return 0.0
        if 0.0 <= score <= 1.0:
            return max(0.0, min(float(score), 1.0))
        return max(0.0, min(float(score) / 100.0, 1.0))
    except Exception:
        return 0.0


def sources_for_cluster(cluster: Dict[str, Any]) -> Counter:
    sents = cluster.get("sentences", []) or []
    srcs = [s.get("source", "unknown") or "unknown" for s in sents]
    return Counter(srcs)


def to_download_blobs(clusters: List[Dict[str, Any]]):
    data = []
    for idx, c in enumerate(clusters, 1):
        rep_text = c.get("representative") or c.get("summary", "")
        data.append(
            {
                "cluster_id": idx,
                "title": generate_cluster_title(rep_text),
                "summary": c.get("summary", ""),
                "reliability": c.get("reliability", 0.0),
                "sources": list(sources_for_cluster(c).items()),
                "sentences": c.get("sentences", []),
            }
        )
    json_blob = json.dumps(data, ensure_ascii=False, indent=2)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["cluster_id", "title", "reliability", "summary", "sentence_text", "source", "date"])
    for item in data:
        for s in item["sentences"]:
            writer.writerow([
                item["cluster_id"],
                item["title"],
                f"{item['reliability']:.1f}",
                item["summary"],
                s.get("text", ""),
                s.get("source", ""),
                s.get("date", "") or "",
            ])
    csv_blob = output.getvalue()
    return json_blob, csv_blob


def filter_sort_clusters(clusters, min_reliability, multi_source_only, sort_by, keyword):
    def keep(c):
        if multi_source_only and len(sources_for_cluster(c)) < 2:
            return False
        if c.get("reliability", 0.0) < min_reliability:
            return False
        if keyword:
            k = keyword.lower()
            if k not in (c.get("summary", "") or "").lower() and k not in (c.get("representative", "") or "").lower():
                return False
        return True

    kept = [c for c in clusters if keep(c)]
    if sort_by == "Reliability":
        kept.sort(key=lambda x: x.get("reliability", 0.0), reverse=True)
    elif sort_by == "Title":
        kept.sort(key=lambda x: generate_cluster_title(x.get("representative", "") or x.get("summary", "")))
    elif sort_by == "Sources":
        kept.sort(key=lambda x: len(sources_for_cluster(x)), reverse=True)
    return kept


def render_cluster_cards(clusters: List[Dict[str, Any]], top_k: int):
    cards = []
    for idx, cluster in enumerate(clusters, 1):
        rep_text = cluster.get("representative") or cluster.get("summary", "")
        title = generate_cluster_title(rep_text)
        reliability = cluster.get("reliability", 0.0)
        src_counts = sources_for_cluster(cluster)
        domain_rels = {d: int(fetch_source_reliability(d) * 100) for d in src_counts.keys()}

        sentence_items = []
        for s in cluster.get("sentences", [])[:top_k]:
            domain = s.get("source", "unknown")
            relp = domain_rels.get(domain, int(fetch_source_reliability(domain) * 100))
            sentence_items.append(
                html.Li(
                    [
                        html.Div(s.get("text", "")),
                        html.Small(f"{domain} | {s.get('date', 'N/A') or 'N/A'} | Source reliability {relp}%"),
                    ],
                    className="mb-2",
                )
            )

        cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.Strong(f"{idx}. {title}"),
                                    dbc.Badge(f"{reliability:.1f}%", color="primary", className="ms-2"),
                                ]
                            ),
                            dbc.Progress(value=reliability, color="success", striped=True, animated=False, className="mt-2"),
                        ]
                    ),
                    dbc.CardBody(
                        [
                            html.Div(cluster.get("summary", ""), className="mb-2"),
                            html.Div(
                                [
                                    html.Span("Sources:", className="text-muted me-2"),
                                    html.Span([
                                        html.Span(f"{d} ({cnt}) | {domain_rels.get(d, 0)}%", className="chip")
                                        for d, cnt in list(src_counts.items())[:8]
                                    ]) if src_counts else html.Small("Sources unavailable"),
                                ]
                            ),
                            html.Div(className="divider"),
                            html.H6(f"Contributing sentences (top {top_k})"),
                            html.Ul(sentence_items, className="mb-0"),
                        ]
                    ),
                ],
                className="mb-3 glass-card",
            )
        )
    return cards


def layout_intro():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Span("Pipeline overview", className="pill"),
                                    html.H3("From noisy articles to objective narratives"),
                                    html.P(
                                        "We fetch diverse sources, group narratives, neutralize bias, and score reliability so you can trust what you read.",
                                        className="text-muted",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.Small("Step 1", className="text-muted"),
                                                        html.H5("Gather & filter"),
                                                        html.P("Pull multi-source articles and keep only relevant, dense sentences.", className="mb-0 text-muted"),
                                                    ]),
                                                    className="glass-card",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.Small("Step 2", className="text-muted"),
                                                        html.H5("Cluster & summarize"),
                                                        html.P("Group narratives, merge overlaps, and summarize clearly.", className="mb-0 text-muted"),
                                                    ]),
                                                    className="glass-card",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        html.Small("Step 3", className="text-muted"),
                                                        html.H5("Neutralize & score"),
                                                        html.P("Rewrite to objective language and score reliability across domains.", className="mb-0 text-muted"),
                                                    ]),
                                                    className="glass-card",
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="gy-2 mt-2",
                                    ),
                                ]
                            ),
                            className="glass-card",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Span("Try these", className="pill"),
                                    html.H5("Quick actions"),
                                    dbc.Button("Run Article Analysis", id="goto-analysis", color="primary", className="w-100 mb-2"),
                                    dbc.Button("Open Objectivity Lab", id="goto-objectivity", color="secondary", className="w-100 mb-2"),
                                    dbc.Button("Explore Utilities", id="goto-utilities", color="info", className="w-100"),
                                    html.Div(className="divider"),
                                    html.P("Best for: comparative news scans, bias reduction, media literacy demos.", className="text-muted mb-1"),
                                    html.Small("Tip: start with 8-12 articles for fast, stable clusters.", className="text-muted"),
                                ]
                            ),
                            className="glass-card",
                        ),
                        md=4,
                    ),
                ],
                className="gy-3",
            ),
        ],
        className="section",
        fluid=True,
    )


def layout_analysis():
    return dbc.Container(
        [
            html.Div(
                [
                    html.Span("Narrative engine", className="pill"),
                    html.H3("Cluster and score news narratives"),
                    html.P(
                        "Fetch diverse articles, auto-cluster narratives, rewrite to neutral language, and export results.",
                        className="text-muted",
                    ),
                ],
                className="mb-2",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Topic or URL"),
                                        dbc.Input(id="topic-input", placeholder="e.g., global economic outlook", type="text"),
                                        html.Small("Use a concise topic or a single article URL.", className="text-muted"),
                                    ],
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Articles to fetch"),
                                        dcc.Slider(id="link-count", min=5, max=20, step=1, value=10, marks=None, tooltip={"placement": "bottom"}),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Summarization detail"),
                                        dcc.Dropdown(
                                            id="summarize-level",
                                            options=[
                                                {"label": "Fast", "value": "fast"},
                                                {"label": "Medium", "value": "medium"},
                                                {"label": "Best (heavier)", "value": "best"},
                                            ],
                                            value="medium",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Sources"),
                                        dbc.Checkbox(id="diverse-links", value=True, className="mt-1", label="Favor diverse domains"),
                                        dbc.Button("Analyze", id="run-analysis", color="primary", className="mt-2 w-100"),
                                        html.Div(id="analysis-status", className="mt-2 text-muted"),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="gy-3",
                        )
                    ]
                ),
                className="glass-card mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Filters"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Min reliability (%)"),
                                            dcc.Slider(
                                                id="min-rel-slider",
                                                min=0,
                                                max=100,
                                                step=5,
                                                value=0,
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Multi-source only"),
                                            dbc.Checklist(
                                                id="multi-source-toggle",
                                                options=[{"label": "Require >=2 sources", "value": "multi"}],
                                                value=[],
                                                switch=True,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Sort by"),
                                            dcc.Dropdown(
                                                id="sort-by",
                                                options=[
                                                    {"label": "Reliability", "value": "Reliability"},
                                                    {"label": "Sources", "value": "Sources"},
                                                    {"label": "Title", "value": "Title"},
                                                ],
                                                value="Reliability",
                                                clearable=False,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Keyword contains"),
                                            dbc.Input(id="keyword-filter", placeholder="Filter summaries"),
                                        ],
                                        md=3,
                                    ),
                                ],
                                className="gy-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Max sentences per narrative"),
                                            dcc.Slider(
                                                id="top-k-slider",
                                                min=3,
                                                max=15,
                                                step=1,
                                                value=5,
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="gy-2 mt-2",
                            ),
                        ]
                    ),
                ],
                className="glass-card mb-3",
            ),
            dcc.Loading(html.Div(id="analysis-results"), type="default"),
            dcc.Store(id="analysis-store"),
            dcc.Download(id="download-json"),
            dcc.Download(id="download-csv"),
        ],
        className="section",
        fluid=True,
    )


def layout_objectivity():
    default_sentence = (
        "The corrupt politician delivered a disastrous speech, infuriating the brave citizens who are fighting for justice."
    )
    return dbc.Container(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Span("Rewrite lab", className="pill"),
                        html.H3("Make subjective text neutral"),
                        html.P("Compare objectivity scores before and after rewriting. Great for quick bias checks.", className="text-muted"),
                        dbc.Textarea(id="objectify-input", value=default_sentence, rows=4, className="mt-2"),
                        dbc.Button("Objectify", id="objectify-button", color="primary", className="mt-3"),
                        html.Small("Tip: try loaded adjectives or charged phrasing.", className="text-muted"),
                        dcc.Loading(html.Div(id="objectify-results", className="mt-4")),
                    ]
                ),
                className="glass-card",
            ),
        ],
        className="section",
        fluid=True,
    )


def layout_utilities():
    return dbc.Container(
        [
            html.Span("Building blocks", className="pill"),
            html.H3("Utilities Explorer"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Synonym Objectivity Ranker"),
                                    html.P("See neutral alternatives ranked by objectivity score.", className="text-muted"),
                                    dbc.Input(id="synonym-word", value="beautiful", placeholder="Enter a word"),
                                    dbc.Button("Rank synonyms", id="synonym-run", color="primary", className="mt-2"),
                                    dcc.Loading(html.Div(id="synonym-results", className="mt-3")),
                                ]
                            ),
                            className="glass-card",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Domain Reliability Checker"),
                                    html.P("Surface the stored reliability score for any news domain.", className="text-muted"),
                                    dbc.Input(id="domain-input", value="nytimes.com", placeholder="cnn.com"),
                                    dbc.Button("Check domain", id="domain-run", color="secondary", className="mt-2"),
                                    dcc.Loading(html.Div(id="domain-results", className="mt-3")),
                                ]
                            ),
                            className="glass-card",
                        ),
                        md=6,
                    ),
                ]
            ),
        ],
        className="section",
        fluid=True,
    )


app: Dash = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Markdown(f"<style>{GLOBAL_STYLE}</style>", dangerously_allow_html=True),
        html.Div(
            [
                dbc.Navbar(
                    [
                        dbc.NavbarBrand("Objective News", class_name="fw-bold text-white"),
                        dbc.Nav(
                            [
                                dbc.NavItem(dbc.NavLink("Intro", href="#", id="nav-intro", class_name="text-white")),
                                dbc.NavItem(dbc.NavLink("Analysis", href="#", id="nav-analysis", class_name="text-white")),
                                dbc.NavItem(dbc.NavLink("Objectivity", href="#", id="nav-objectivity", class_name="text-white")),
                                dbc.NavItem(dbc.NavLink("Utilities", href="#", id="nav-utilities", class_name="text-white")),
                            ],
                            class_name="ms-auto",
                        ),
                    ],
                    color="dark",
                    dark=True,
                    class_name="rounded-3 my-3 px-4",
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.Span("Objective-first ML lab", className="pill"),
                                        html.H2("A modern dashboard for objective news analysis", className="mt-2"),
                                        html.P(
                                            "Analyze narratives, neutralize language, and score reliability across sources—all in one place.",
                                            className="text-muted",
                                        ),
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button("Start Analysis", id="goto-analysis", color="primary"),
                                                dbc.Button("Try Objectivity", id="goto-objectivity", color="secondary"),
                                                dbc.Button("Explore Utilities", id="goto-utilities", color="info"),
                                            ],
                                            className="mt-2",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="hero-card glass-card mb-3",
                ),
                dcc.Tabs(
                    id="main-tabs",
                    value="intro",
                    className="dash-tabs",
                    children=[
                        dcc.Tab(label="Introduction", value="intro"),
                        dcc.Tab(label="Article Analysis", value="analysis"),
                        dcc.Tab(label="Objectivity Playground", value="objectivity"),
                        dcc.Tab(label="Utilities", value="utilities"),
                    ],
                ),
                html.Div(id="tab-content", className="mt-3"),
            ],
            className="page-shell",
        ),
    ],
    fluid=True,
)

# Provide full component tree for callback validation to avoid "nonexistent object" warnings
app.validation_layout = html.Div([
    app.layout,
    layout_intro(),
    layout_analysis(),
    layout_objectivity(),
    layout_utilities(),
])


@app.callback(Output("main-tabs", "value"), Input("url", "search"))
def sync_tabs_from_url(search):
    qs = parse_qs((search or "").lstrip("?"))
    page = (qs.get("page", ["intro"]) or ["intro"])[0]
    return page if page in {"intro", "analysis", "objectivity", "utilities"} else "intro"


@app.callback(
    Output("url", "search"),
    [
        Input("goto-analysis", "n_clicks"),
        Input("goto-objectivity", "n_clicks"),
        Input("goto-utilities", "n_clicks"),
        Input("nav-intro", "n_clicks"),
        Input("nav-analysis", "n_clicks"),
        Input("nav-objectivity", "n_clicks"),
        Input("nav-utilities", "n_clicks"),
        Input("main-tabs", "value"),
    ],
    prevent_initial_call=True,
)
def update_url_from_nav(to_analysis, to_objectivity, to_utilities, nav_intro, nav_analysis, nav_objectivity, nav_utilities, tab_value):
    trigger = ctx.triggered_id
    page_by_trigger = {
        "goto-analysis": "analysis",
        "nav-analysis": "analysis",
        "goto-objectivity": "objectivity",
        "nav-objectivity": "objectivity",
        "goto-utilities": "utilities",
        "nav-utilities": "utilities",
        "nav-intro": "intro",
    }
    if trigger == "main-tabs":
        target = tab_value or "intro"
    else:
        target = page_by_trigger.get(trigger, "intro")
    return f"?page={target}"


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "analysis":
        return layout_analysis()
    if tab == "objectivity":
        return layout_objectivity()
    if tab == "utilities":
        return layout_utilities()
    return layout_intro()


@app.callback(
    [Output("analysis-store", "data"), Output("analysis-status", "children")],
    Input("run-analysis", "n_clicks"),
    [State("topic-input", "value"), State("link-count", "value"), State("summarize-level", "value"), State("diverse-links", "value")],
    prevent_initial_call=True,
)
def run_analysis(n_clicks, topic, link_count, summarize_level, diverse_links):
    if not n_clicks:
        return no_update, no_update
    if not topic:
        return no_update, "Please enter a topic or URL."
    try:
        level = {"best": "slow"}.get(summarize_level, summarize_level)
        result = article_analysis(
            text=topic,
            link_n=int(link_count or 10),
            diverse_links=bool(diverse_links),
            summarize_level=level,
            progress_callback=None,
        )
        serialized = [serialize_cluster(c) for c in (result or [])]
        if not serialized:
            return [], "No narratives found. Try a different query."
        return serialized, f"Analysis complete. {len(serialized)} narratives found."
    except Exception as e:
        logger.exception("Analysis failed")
        return no_update, f"Error during analysis: {e}"


@app.callback(
    Output("analysis-results", "children"),
    [
        Input("analysis-store", "data"),
        Input("min-rel-slider", "value"),
        Input("multi-source-toggle", "value"),
        Input("sort-by", "value"),
        Input("keyword-filter", "value"),
        Input("top-k-slider", "value"),
        Input("main-tabs", "value"),
    ],
)
def render_analysis_results(store_data, min_rel, multi_toggle, sort_by, keyword, top_k, current_tab):
    if current_tab != "analysis":
        return no_update
    if not store_data:
        return dbc.Alert("Run an analysis to see results.", color="info")

    multi_only = "multi" in (multi_toggle or [])
    filtered = filter_sort_clusters(store_data, float(min_rel or 0.0), multi_only, sort_by or "Reliability", keyword or "")
    metrics = []
    sources = Counter()
    for c in store_data:
        sources.update(sources_for_cluster(c))
    avg_rel = sum(c.get("reliability", 0.0) for c in store_data) / len(store_data) if store_data else 0.0
    metrics.append(dbc.Col(dbc.Card([dbc.CardHeader("Narratives"), dbc.CardBody(html.H3(len(store_data)))], className="glass-card metric-card")))
    metrics.append(dbc.Col(dbc.Card([dbc.CardHeader("Distinct Sources"), dbc.CardBody(html.H3(len(sources)))], className="glass-card metric-card")))
    metrics.append(dbc.Col(dbc.Card([dbc.CardHeader("Avg Reliability"), dbc.CardBody(html.H3(f"{avg_rel:.1f}%"))], className="glass-card metric-card")))

    cards = render_cluster_cards(filtered, top_k=int(top_k or 5))

    return dbc.Container(
        [
            dbc.Row(metrics, className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Download JSON", id="download-json-btn", color="secondary", className="me-2"), width="auto"),
                    dbc.Col(dbc.Button("Download CSV", id="download-csv-btn", color="secondary"), width="auto"),
                ],
                className="mb-2",
            ),
            dbc.Alert("Tune filters to focus on multi-source, higher-confidence narratives. Then export your view.", color="light", className="glass-card"),
            html.Div(cards),
        ]
    )


@app.callback(Output("download-json", "data"), Input("download-json-btn", "n_clicks"), State("analysis-store", "data"), prevent_initial_call=True)
def download_json(n_clicks, data):
    if not n_clicks or not data:
        return no_update
    return dict(content=json.dumps(data, indent=2), filename="analysis.json")


@app.callback(Output("download-csv", "data"), Input("download-csv-btn", "n_clicks"), State("analysis-store", "data"), prevent_initial_call=True)
def download_csv(n_clicks, data):
    if not n_clicks or not data:
        return no_update
    _, csv_blob = to_download_blobs(data)
    return dict(content=csv_blob, filename="analysis.csv", type="text/csv")


@app.callback(Output("objectify-results", "children"), Input("objectify-button", "n_clicks"), State("objectify-input", "value"), prevent_initial_call=True)
def handle_objectify(n_clicks, text):
    if not text:
        return dbc.Alert("Please enter a sentence first.", color="warning")
    original_score = calculate_objectivity(text)
    objectified = objectify_text(text)
    new_score = calculate_objectivity(objectified)
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card([dbc.CardHeader("Original Score"), dbc.CardBody(html.H4(f"{original_score:.3f}"))])),
                    dbc.Col(dbc.Card([dbc.CardHeader("New Score"), dbc.CardBody(html.H4(f"{new_score:.3f}"))])),
                    dbc.Col(dbc.Card([dbc.CardHeader("Delta"), dbc.CardBody(html.H4(f"{(new_score - original_score):+.3f}"))])),
                ],
                className="mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Final Neutral Text"),
                    dbc.CardBody(html.Div(objectified)),
                ]
            ),
        ]
    )


@app.callback(Output("synonym-results", "children"), Input("synonym-run", "n_clicks"), State("synonym-word", "value"), prevent_initial_call=True)
def handle_synonyms(n_clicks, word):
    if not word:
        return dbc.Alert("Please enter a word to analyze.", color="warning")
    synonyms_raw = get_synonyms(word, deep=True, include_external=True, topn=15)
    if not synonyms_raw:
        return dbc.Alert(f"No synonyms found for '{word}'.", color="info")
    synonym_data = sorted(
        [
            {
                "word": syn["word"],
                "definition": syn.get("definition"),
                "objectivity": calculate_objectivity(syn["word"]),
            }
            for syn in synonyms_raw
        ],
        key=lambda x: x["objectivity"],
        reverse=True,
    )
    rows = []
    for i, syn in enumerate(synonym_data, start=1):
        rows.append(
            dbc.ListGroupItem(
                [
                    html.Strong(f"{i}. {syn['word']}"),
                    html.Span(f"Objectivity: {syn['objectivity']:.3f}", className="float-end"),
                    html.Div(syn.get("definition") or ""),
                ]
            )
        )
    return dbc.ListGroup(rows)


@app.callback(Output("domain-results", "children"), Input("domain-run", "n_clicks"), State("domain-input", "value"), prevent_initial_call=True)
def handle_domain(n_clicks, domain):
    if not domain:
        return dbc.Alert("Please enter a domain.", color="warning")
    normalized = normalize_domain(domain)
    score, _ = get_source_label(normalized, _load_source_df())
    if score is None:
        return dbc.Alert(f"No score found for {normalized}.", color="info")
    percent = ((score + 1) / 2 * 100)
    return dbc.Card(
        [
            dbc.CardHeader(f"Reliability for {normalized}"),
            dbc.CardBody(
                [
                    html.Div(f"Reliability score [-1 to 1]: {score:.3f}" if score is not None else "No score found"),
                    html.Div(f"Reliability percentage: {percent:.1f}%"),
                    dbc.Progress(value=percent, className="mt-2"),
                ]
            ),
        ]
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
