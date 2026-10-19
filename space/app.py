"""Hugging Face Space: aspect-level sentiment for car reviews, with the evidence behind every label."""

import html
import os
import sys
from collections import Counter
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from review_intel.infer import Analyzer  # noqa: E402
from review_intel.schema import ASPECTS, NICE  # noqa: E402

MODEL = os.environ.get("MODEL_ID", "Phoenixak99/car-review-absa")
analyzer = Analyzer(MODEL, threads=int(os.environ.get("THREADS", "2")))
analyzer.analyze("Warm-up: the ride is smooth and the screen is great.")  # load both models before the first visitor

EXAMPLES = [
    "Bought a 2022 Honda CR-V last spring. It's roomy and the seats are great on long drives, and we average about 34 mpg. "
    "The touchscreen, though, freezes at least once a week and Apple CarPlay drops out constantly.",
    "Third transmission in 60k miles. The dealer keeps saying it's normal. Looks great in the driveway, I'll give it that, "
    "but I would never buy another one.",
    "Fast, sharp handling and the brakes are amazing. Insurance on it is brutal though.",
    "The Model Y's range is fine for my commute but in winter it drops a lot, and the ride is stiff over potholes. "
    "Autopilot and the safety features make long highway drives much less tiring.",
]
COLORS = {"positive": "#2E7D4F", "negative": "#C2341B", "mixed": "#A8873E"}
ENTITY_COLORS = {"car make": "#141310", "car model": "#141310", "car part": "#6B665C", "place": "#6B665C"}


def verdict_html(a) -> str:
    chips = "".join(
        f"<span class='chip' style='--c:{COLORS[x.sentiment]};border-color:{COLORS[x.sentiment]}'><b>{NICE[x.aspect]}</b> {x.sentiment}"
        f"<i>{x.confidence:.0%}</i></span>"
        for x in sorted(a.aspects, key=lambda x: -x.confidence)
    )
    if not chips:
        chips = "<span class='muted'>No specific aspect was discussed.</span>"
    return (
        f"<div class='verdict'><p class='kicker'>Overall</p><p class='overall' style='--c:{COLORS[a.overall]}'>"
        f"{a.overall} <span class='muted'>{a.overall_confidence:.0%}</span></p>"
        f"<p class='kicker'>Aspects</p><div class='chips'>{chips}</div></div>"
    )


def review_html(a) -> str:
    """The review with every sentence that drives an aspect underlined, and entities marked."""
    by_sentence: dict[int, list] = {}
    for x in a.aspects:
        for s in x.evidence:
            by_sentence.setdefault(s, []).append(x)
    pos, parts = 0, []
    for i, sent in enumerate(a.sentences):
        start = a.text.find(sent, pos)
        pos = start + len(sent)
        inner = mark_entities(sent, [e for e in a.entities if start <= e["start"] < pos], start)
        tags = by_sentence.get(i, [])
        if tags:
            color = COLORS[tags[0].sentiment]
            labels = " · ".join(f"{NICE[t.aspect]} {t.sentiment}" for t in tags)
            parts.append(
                f"<span class='ev' style='--c:{color}' title='{html.escape(labels)}'>{inner}"
                f"<sup>{html.escape(labels)}</sup></span>"
            )
        else:
            parts.append(inner)
    return "<div class='review'>" + " ".join(parts) + "</div>"


def mark_entities(sentence: str, ents: list[dict], offset: int) -> str:
    out, last = [], 0
    for e in sorted(ents, key=lambda e: e["start"]):
        s, t = e["start"] - offset, e["end"] - offset
        if s < last:
            continue
        out.append(html.escape(sentence[last:s]))
        out.append(
            f"<mark style='--c:{ENTITY_COLORS.get(e['label'], '#141310')}'>{html.escape(sentence[s:t])}"
            f"<small>{e['label']}</small></mark>"
        )
        last = t
    out.append(html.escape(sentence[last:]))
    return "".join(out)


def analyse(text: str):
    text = (text or "").strip()
    if len(text) < 10:
        return "<p class='muted'>Paste a review (at least a sentence).</p>", ""
    a = analyzer.analyze(text)
    return verdict_html(a), review_html(a)


def analyse_csv(file, column: str):
    if file is None:
        return None, None, "<p class='muted'>Upload a CSV with one review per row.</p>"
    df = pd.read_csv(file.name if hasattr(file, "name") else file)
    col = column.strip() or next((c for c in df.columns if df[c].dtype == object), None)
    if col not in df.columns:
        raise gr.Error(f"No column called '{col}'. Columns: {', '.join(map(str, df.columns))}")
    texts = df[col].dropna().astype(str).tolist()[:300]
    analyzer.use_entities = False
    try:
        results = [r for i in range(0, len(texts), 16) for r in analyzer.analyze_batch(texts[i : i + 16])]
    finally:
        analyzer.use_entities = True
    counts = {a: Counter() for a in ASPECTS}
    praise, complaints = [], []
    for r in results:
        for x in r.aspects:
            counts[x.aspect][x.sentiment] += 1
            for s in x.evidence[:1]:
                (praise if x.sentiment == "positive" else complaints if x.sentiment == "negative" else []).append(
                    (x.confidence, NICE[x.aspect], r.sentences[s])
                )
    fig = go.Figure()
    for sent in ("positive", "mixed", "negative"):
        fig.add_bar(
            name=sent,
            y=[NICE[a] for a in ASPECTS],
            x=[counts[a][sent] for a in ASPECTS],
            orientation="h",
            marker_color=COLORS[sent],
        )
    fig.update_layout(
        barmode="stack",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#FFFDF8",
        plot_bgcolor="#FFFDF8",
        font=dict(color="#141310"),
        legend=dict(orientation="h", y=1.12),
        title=f"{len(results)} reviews · aspect mentions",
    )
    table = pd.DataFrame(
        [
            {
                "review": r.text[:140],
                "overall": r.overall,
                **{NICE[a]: next((x.sentiment for x in r.aspects if x.aspect == a), "") for a in ASPECTS},
            }
            for r in results
        ]
    )

    def top(items):
        items = sorted(items, reverse=True)[:5]
        return (
            "".join(f"<li><b>{html.escape(a)}</b> — {html.escape(s)}</li>" for _, a, s in items)
            or "<li class='muted'>none</li>"
        )

    summary = (
        f"<div class='two'><div><p class='kicker'>Top praise</p><ul>{top(praise)}</ul></div>"
        f"<div><p class='kicker'>Top complaints</p><ul>{top(complaints)}</ul></div></div>"
    )
    return fig, table, summary


CSS = (HERE / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="Car Review Intelligence") as demo:
    gr.HTML(
        "<header class='hd'><p class='kicker'>Review intelligence v2 · 8 aspects · evidence · entities</p>"
        "<h1>What did the owner actually say?</h1><p class='lede'>A fine-tuned ModernBERT reads a car review and "
        "returns the reviewer's opinion on eight aspects — price, performance, comfort, reliability, design, safety, "
        "fuel economy and technology — with the sentences behind each label underlined. GLiNER marks the cars and parts "
        "mentioned. Runs on 2 CPU cores in well under a second.</p>"
        "<p class='links'><a href='https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs' target='_blank'>Code, data & eval ↗</a>"
        f"<a href='https://huggingface.co/{MODEL}' target='_blank'>Model card ↗</a>"
        "<a href='https://ahmed.fynxforge.com' target='_blank'>Ahmed Khaled Ali ↗</a></p></header>"
    )
    with gr.Tab("Analyse a review"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                box = gr.Textbox(label="Review", lines=7, placeholder="Paste an owner review of any car…")
                btn = gr.Button("Analyse", variant="primary")
                gr.HTML("<p class='muted try'>Or try one:</p>")
                gr.Examples(EXAMPLES, inputs=box, label=None)
            with gr.Column(scale=6):
                verdict = gr.HTML("<p class='muted'>Results appear here.</p>")
                review = gr.HTML()
        btn.click(analyse, box, [verdict, review])
        box.submit(analyse, box, [verdict, review])
    with gr.Tab("Analyse a CSV"):
        with gr.Row():
            up = gr.File(label="CSV file (up to 300 rows are analysed)", file_types=[".csv"])
            col = gr.Textbox(
                label="Column with the review text", placeholder="review (leave empty to use the first text column)"
            )
        run = gr.Button("Analyse file", variant="primary")
        summary = gr.HTML()
        chart = gr.Plot()
        table = gr.Dataframe(wrap=True)
        run.click(analyse_csv, [up, col], [chart, table, summary])
    gr.HTML(
        "<footer class='ft'>The model was trained on synthetic owner reviews generated and double-checked by a "
        "local Qwen3-8B teacher (real car names, invented opinions) — see the repository for how and why. "
        "Entities: urchade/gliner_small-v2.1 (Apache-2.0). First load after the Space sleeps takes about a minute.</footer>"
    )

demo.queue(default_concurrency_limit=2, max_size=20)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base())
