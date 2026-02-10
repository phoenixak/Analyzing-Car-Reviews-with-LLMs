"""
Car Reviews Analyzer - Hugging Face Demo
A comprehensive NLP analysis tool for automotive customer feedback.

This demo showcases multiple NLP capabilities:
- Sentiment Analysis with DistilBERT
- Topic Modeling with BERTopic
- Translation (EN to ES)
- Question Answering
- Text Summarization
- Named Entity Recognition
- Aspect-Based Sentiment Analysis
"""

from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import plotly.express as px
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configure device
device = 0 if torch.cuda.is_available() else -1

# Global models cache
models_cache: Dict[str, object] = {}


def get_model(key: str):
    """Lazily load and cache an NLP model by key.

    Models are loaded only on first use and cached for subsequent calls.
    """
    if key not in models_cache:
        try:
            if key == "sentiment":
                print("Loading Sentiment Analysis model...")
                models_cache["sentiment"] = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device,
                )
            elif key == "translation":
                print("Loading Translation model...")
                models_cache["translation"] = pipeline(
                    "translation_en_to_es",
                    model="Helsinki-NLP/opus-mt-en-es",
                    device=device,
                )
            elif key == "qa":
                print("Loading QA model...")
                models_cache["qa"] = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=device,
                )
            elif key == "summarization":
                print("Loading Summarization model...")
                models_cache["summarization"] = pipeline(
                    "summarization",
                    model="t5-small",
                    device=device,
                )
            elif key == "ner":
                print("Loading NER model...")
                models_cache["ner"] = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=device,
                )
            else:
                raise ValueError(f"Unknown model key: {key}")
            print(f"Model '{key}' loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{key}': {e}. "
                "Please check your internet connection and try again."
            ) from e
    return models_cache[key]


def analyze_sentiment(text: str) -> Tuple[str, float, str]:
    """Analyze the sentiment of input text using DistilBERT.

    Returns:
        Tuple of (label, confidence percentage, HTML progress bar).
    """
    if not text or not text.strip():
        return "No input", 0.0, "<p>Please enter text to analyze.</p>"

    try:
        classifier = get_model("sentiment")
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]
        confidence_pct = round(score * 100, 2)

        if label == "POSITIVE":
            color = "#28a745"
            icon = "\u2705"
        else:
            color = "#dc3545"
            icon = "\u274c"

        progress_html = f"""
        <div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #333;">
            <h3 style="margin: 0 0 15px 0; color: #e0e0e0;">{icon} Sentiment Result</h3>
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                <span style="font-size: 1.4em; font-weight: bold; color: {color};">{label}</span>
                <span style="color: #aaa; font-size: 1.1em;">({confidence_pct}% confidence)</span>
            </div>
            <div style="background: #333; border-radius: 8px; height: 24px; overflow: hidden;">
                <div style="width: {confidence_pct}%; height: 100%; background: {color}; border-radius: 8px; transition: width 0.5s;"></div>
            </div>
        </div>
        """
        return label, confidence_pct, progress_html

    except Exception as e:
        return "Error", 0.0, f"<p style='color:red;'>Error: {e}</p>"


def extract_topics(texts: List[str], num_topics: int = 5) -> Tuple[str, str]:
    """Extract topics from a list of texts using BERTopic.

    Returns:
        Tuple of (topics summary HTML, Plotly figure or message).
    """
    if not texts or all(not t.strip() for t in texts):
        return "<p>Please enter texts to analyze.</p>", None

    try:
        # Filter out empty strings
        texts = [t.strip() for t in texts if t.strip()]

        if len(texts) < 3:
            return "<p>Please provide at least 3 texts for topic modeling.</p>", None

        # Configure BERTopic for small datasets
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        umap_model = UMAP(
            n_neighbors=min(3, len(texts) - 1),
            n_components=min(2, len(texts) - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=max(2, len(texts) // 5),
            min_samples=1,
            metric="euclidean",
            prediction_data=True,
        )

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=num_topics if num_topics > 0 else "auto",
            verbose=False,
        )

        topics, probs = topic_model.fit_transform(texts)

        # Get topic info
        try:
            topic_info = topic_model.get_topic_info()
        except Exception:
            topic_info = pd.DataFrame(columns=["Topic", "Count", "Name"])

        # Build HTML summary
        html_parts = [
            '<div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #333;">',
            '<h3 style="color: #e0e0e0; margin: 0 0 15px 0;">Discovered Topics</h3>',
        ]

        for _, row in topic_info.iterrows():
            topic_id = row.get("Topic", "?")
            count = row.get("Count", 0)
            name = row.get("Name", "Unknown")

            if topic_id == -1:
                topic_label = "Outliers"
            else:
                topic_label = f"Topic {topic_id}"
                # Get top words for this topic
                try:
                    top_words = topic_model.get_topic(topic_id)
                    if top_words:
                        word_list = ", ".join([w for w, _ in top_words[:5]])
                        topic_label += f": {word_list}"
                except Exception:
                    pass

            html_parts.append(
                f'<div style="padding: 10px; margin: 8px 0; background: #2a2a4a; border-radius: 8px; border-left: 4px solid #4a90d9;">'
                f'<strong style="color: #4a90d9;">{topic_label}</strong>'
                f'<span style="color: #aaa; margin-left: 10px;">({count} documents)</span>'
                f"</div>"
            )

        html_parts.append("</div>")
        topics_html = "\n".join(html_parts)

        # Create visualization
        fig = None
        try:
            topic_counts = pd.DataFrame(
                {"Topic": topics}
            ).value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]
            topic_counts["Topic"] = topic_counts["Topic"].astype(str)
            fig = px.bar(
                topic_counts,
                x="Topic",
                y="Count",
                title="Topic Distribution",
                color="Topic",
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
        except Exception:
            pass

        return topics_html, fig

    except Exception as e:
        return f"<p style='color:red;'>Error in topic modeling: {e}</p>", None


def translate_text(text: str) -> str:
    """Translate text from English to Spanish using Helsinki-NLP/opus-mt-en-es."""
    if not text or not text.strip():
        return "Please enter text to translate."

    try:
        translator = get_model("translation")
        result = translator(text, max_length=512)
        translated = result[0]["translation_text"]

        output = (
            f"Original (EN):\n{text}\n\n"
            f"Translation (ES):\n{translated}"
        )
        return output

    except Exception as e:
        return f"Translation error: {e}"


def answer_question(context: str, question: str) -> str:
    """Answer a question given a context passage using extractive QA."""
    if not context or not context.strip():
        return "Please provide a context passage."
    if not question or not question.strip():
        return "Please provide a question."

    try:
        qa_model = get_model("qa")
        result = qa_model(question=question, context=context)

        answer = result["answer"]
        score = round(result["score"] * 100, 2)

        output = (
            f"Answer: {answer}\n"
            f"Confidence: {score}%\n\n"
            f"Context excerpt: ...{context[max(0, result['start']-30):result['end']+30]}..."
        )
        return output

    except Exception as e:
        return f"QA error: {e}"


def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarize text using T5-small."""
    if not text or not text.strip():
        return "Please enter text to summarize."

    try:
        summarizer = get_model("summarization")

        # T5 needs minimum length
        min_length = min(30, max_length // 3)
        input_length = len(text.split())

        if input_length < 20:
            return "Input text is too short to summarize. Please provide a longer passage."

        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        summary = result[0]["summary_text"]

        compression = round((1 - len(summary.split()) / input_length) * 100, 1)

        output = (
            f"Summary:\n{summary}\n\n"
            f"Compression ratio: {compression}% reduction\n"
            f"Original: {input_length} words | Summary: {len(summary.split())} words"
        )
        return output

    except Exception as e:
        return f"Summarization error: {e}"


def extract_entities(text: str) -> str:
    """Extract named entities from text using BERT-large NER."""
    if not text or not text.strip():
        return "Please enter text to analyze."

    try:
        ner_model = get_model("ner")
        results = ner_model(text)

        if not results:
            return "No named entities found in the text."

        # Group entities by type
        entity_groups: Dict[str, List[str]] = {}
        for ent in results:
            group = ent.get("entity_group", "UNKNOWN")
            word = ent.get("word", "")
            score = ent.get("score", 0)

            if group not in entity_groups:
                entity_groups[group] = []
            entity_groups[group].append(f"{word} ({round(score * 100, 1)}%)")

        # Format output
        entity_labels = {
            "PER": "Person",
            "ORG": "Organization",
            "LOC": "Location",
            "MISC": "Miscellaneous",
        }

        output_parts = ["Named Entities Found:\n"]
        for group, entities in entity_groups.items():
            label = entity_labels.get(group, group)
            output_parts.append(f"\n{label}:")
            for entity in entities:
                output_parts.append(f"  - {entity}")

        return "\n".join(output_parts)

    except Exception as e:
        return f"NER error: {e}"


def analyze_aspect_sentiment(text: str) -> str:
    """Perform aspect-based sentiment analysis on car review text.

    Detects automotive aspects via keyword matching, extracts relevant sentences,
    and runs sentiment analysis on each aspect.
    """
    if not text or not text.strip():
        return "Please enter a car review to analyze."

    try:
        classifier = get_model("sentiment")

        # Define automotive aspects and their keywords
        aspects = {
            "Performance": ["engine", "power", "horsepower", "acceleration", "speed", "torque", "performance", "fast", "slow", "turbo"],
            "Comfort": ["comfort", "comfortable", "seat", "seats", "ride", "smooth", "suspension", "cabin", "quiet", "noise"],
            "Fuel Efficiency": ["fuel", "gas", "mileage", "mpg", "economy", "efficient", "consumption", "hybrid", "electric"],
            "Safety": ["safety", "safe", "airbag", "brake", "brakes", "crash", "collision", "abs", "stability", "assist"],
            "Technology": ["technology", "tech", "screen", "display", "infotainment", "bluetooth", "gps", "navigation", "camera", "sensor"],
            "Design": ["design", "style", "look", "looks", "exterior", "interior", "color", "sleek", "modern", "aesthetic"],
            "Reliability": ["reliability", "reliable", "durable", "maintenance", "repair", "warranty", "quality", "build", "issue", "problem"],
            "Value": ["price", "value", "cost", "worth", "expensive", "cheap", "affordable", "deal", "money", "budget"],
        }

        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        aspect_results = {}

        for aspect, keywords in aspects.items():
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(kw in sentence_lower for kw in keywords):
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                combined = ". ".join(relevant_sentences)
                result = classifier(combined)[0]
                aspect_results[aspect] = {
                    "label": result["label"],
                    "score": round(result["score"] * 100, 2),
                    "evidence": relevant_sentences[:2],
                }

        if not aspect_results:
            return "No specific automotive aspects detected in the text. Try including keywords related to performance, comfort, safety, technology, etc."

        # Format output
        output_parts = ["Aspect-Based Sentiment Analysis:\n"]
        output_parts.append("=" * 45)

        for aspect, data in aspect_results.items():
            sentiment_icon = "\u2705" if data["label"] == "POSITIVE" else "\u274c"
            output_parts.append(
                f"\n{sentiment_icon} {aspect}: {data['label']} ({data['score']}%)"
            )
            for ev in data["evidence"]:
                output_parts.append(f"   \"{ev}\"")

        # Overall summary
        positive_count = sum(1 for d in aspect_results.values() if d["label"] == "POSITIVE")
        total_count = len(aspect_results)
        output_parts.append(f"\n{'=' * 45}")
        output_parts.append(
            f"Overall: {positive_count}/{total_count} aspects positive"
        )

        return "\n".join(output_parts)

    except Exception as e:
        return f"Aspect analysis error: {e}"


def create_sample_data() -> pd.DataFrame:
    """Create sample car review data for demonstration."""
    reviews = [
        {
            "id": 1,
            "review": "The new Tesla Model 3 has incredible acceleration and the autopilot feature is impressive. However, the interior quality could be better for the price point. The touchscreen interface takes some getting used to.",
            "car": "Tesla Model 3",
            "rating": 4,
        },
        {
            "id": 2,
            "review": "I've been driving my Toyota Camry for 6 months and it's been extremely reliable. Great fuel economy and comfortable seats for long drives. The safety features like lane departure warning work flawlessly.",
            "car": "Toyota Camry",
            "rating": 5,
        },
        {
            "id": 3,
            "review": "The BMW M3 delivers outstanding performance with its twin-turbo engine. The handling is razor-sharp and the exhaust note is thrilling. Maintenance costs are high but the driving experience is worth every penny.",
            "car": "BMW M3",
            "rating": 5,
        },
        {
            "id": 4,
            "review": "Disappointed with my Ford Explorer. The transmission has been jerky since day one and the infotainment system frequently freezes. The dealer service has been unhelpful. Would not recommend this model.",
            "car": "Ford Explorer",
            "rating": 2,
        },
        {
            "id": 5,
            "review": "The Honda Civic offers excellent value for money. Fuel efficient, reliable, and packed with technology features. The Honda Sensing suite makes highway driving much less stressful. Perfect commuter car.",
            "car": "Honda Civic",
            "rating": 4,
        },
    ]
    return pd.DataFrame(reviews)


# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.main-header {
    text-align: center;
    padding: 30px 20px;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid #444;
}

.main-header h1 {
    font-size: 2.2em;
    margin: 0 0 10px 0;
    background: linear-gradient(90deg, #4a90d9, #67b26f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #bbb;
    font-size: 1.1em;
    margin: 0;
}

.tab-content {
    padding: 15px;
}

.footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 1px solid #333;
    color: #888;
}

.info-banner {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 15px;
    color: #aaa;
    font-size: 0.95em;
}

.model-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 15px;
    margin: 8px 0;
}

.model-card h4 {
    color: #4a90d9;
    margin: 0 0 8px 0;
}

.model-card p {
    color: #aaa;
    margin: 0;
    font-size: 0.9em;
}

@media (prefers-color-scheme: light) {
    .main-header {
        background: linear-gradient(135deg, #e8eaf6, #c5cae9, #e8eaf6);
        border: 1px solid #ccc;
    }

    .main-header h1 {
        background: linear-gradient(90deg, #1565c0, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #555;
    }

    .info-banner {
        background: linear-gradient(135deg, #e3f2fd, #f1f8e9);
        border: 1px solid #ccc;
        color: #555;
    }

    .model-card {
        background: #f5f5f5;
        border: 1px solid #ddd;
    }

    .model-card h4 {
        color: #1565c0;
    }

    .model-card p {
        color: #666;
    }
}
"""


def create_demo() -> gr.Blocks:
    """Create the full Gradio Blocks demo with all NLP tabs."""

    sample_df = create_sample_data()

    with gr.Blocks(css=custom_css, title="Car Reviews Analyzer") as demo:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>\U0001F697 Car Reviews Analyzer</h1>
            <p>Comprehensive NLP Analysis for Automotive Customer Feedback</p>
        </div>
        """)

        # Info banner (replaces old initialize button)
        gr.HTML("""
        <div class="info-banner">
            \u2139\ufe0f Models load automatically on first use. The first analysis in each tab may take a moment while the model downloads.
        </div>
        """)

        # ---- Tab 1: Sentiment Analysis ----
        with gr.Tab("\U0001F4CA Sentiment Analysis"):
            gr.Markdown("### Analyze the sentiment of car reviews using DistilBERT")

            with gr.Row():
                with gr.Column(scale=2):
                    sentiment_input = gr.Textbox(
                        label="Enter a car review",
                        placeholder="Type or paste a car review here...",
                        lines=4,
                    )
                    sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
                with gr.Column(scale=1):
                    sentiment_label = gr.Textbox(label="Sentiment", interactive=False)
                    sentiment_score = gr.Number(label="Confidence (%)", interactive=False)

            sentiment_html = gr.HTML(label="Result")

            sentiment_btn.click(
                fn=analyze_sentiment,
                inputs=[sentiment_input],
                outputs=[sentiment_label, sentiment_score, sentiment_html],
            )

            gr.Examples(
                examples=[
                    ["The new Tesla Model 3 has incredible acceleration and the autopilot feature is impressive."],
                    ["Disappointed with my Ford Explorer. The transmission has been jerky since day one."],
                    ["The Honda Civic offers excellent value for money. Fuel efficient and reliable."],
                    ["The paint started chipping after just 6 months. Terrible build quality."],
                    ["Absolutely love the BMW M3. The handling is razor-sharp and thrilling to drive."],
                ],
                inputs=[sentiment_input],
            )

        # ---- Tab 2: Topic Modeling ----
        with gr.Tab("\U0001F4D1 Topic Modeling"):
            gr.Markdown("### Discover topics across multiple car reviews using BERTopic")

            topic_input = gr.Textbox(
                label="Enter multiple reviews (one per line)",
                placeholder="Paste multiple car reviews, each on a new line...",
                lines=8,
            )
            with gr.Row():
                num_topics = gr.Slider(
                    minimum=2, maximum=10, value=5, step=1, label="Number of Topics"
                )
                topic_btn = gr.Button("Extract Topics", variant="primary")

            topic_html = gr.HTML(label="Topics")
            topic_plot = gr.Plot(label="Topic Distribution")

            def topic_wrapper(text, n_topics):
                if not text or not text.strip():
                    return "<p>Please enter texts.</p>", None
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                return extract_topics(lines, int(n_topics))

            topic_btn.click(
                fn=topic_wrapper,
                inputs=[topic_input, num_topics],
                outputs=[topic_html, topic_plot],
            )

            gr.Examples(
                examples=[
                    [
                        "The engine performance is outstanding with great horsepower.\n"
                        "Fuel economy is excellent, getting 35 mpg on the highway.\n"
                        "The safety features include lane assist and automatic braking.\n"
                        "Interior comfort is superb with heated leather seats.\n"
                        "The infotainment system has a responsive touchscreen display.\n"
                        "Maintenance costs have been surprisingly low for a luxury car.\n"
                        "The suspension handles bumps smoothly on rough roads."
                    ],
                ],
                inputs=[topic_input],
            )

        # ---- Tab 3: Translation ----
        with gr.Tab("\U0001F30D Translation (EN \u2192 ES)"):
            gr.Markdown("### Translate car reviews from English to Spanish")

            with gr.Row():
                with gr.Column():
                    translate_input = gr.Textbox(
                        label="English Text",
                        placeholder="Enter English text to translate...",
                        lines=4,
                    )
                    translate_btn = gr.Button("Translate", variant="primary")
                with gr.Column():
                    translate_output = gr.Textbox(
                        label="Translation Result",
                        lines=6,
                        interactive=False,
                    )

            translate_btn.click(
                fn=translate_text,
                inputs=[translate_input],
                outputs=[translate_output],
            )

            gr.Examples(
                examples=[
                    ["The car has excellent fuel efficiency and a comfortable interior."],
                    ["I would not recommend this vehicle due to frequent mechanical issues."],
                    ["The safety features and advanced technology make this a great family car."],
                ],
                inputs=[translate_input],
            )

        # ---- Tab 4: Question Answering ----
        with gr.Tab("\u2753 Question Answering"):
            gr.Markdown("### Ask questions about car reviews using extractive QA")

            qa_context = gr.Textbox(
                label="Context (Car Review)",
                placeholder="Paste a car review as context...",
                lines=5,
            )
            qa_question = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the review...",
                lines=2,
            )
            qa_btn = gr.Button("Get Answer", variant="primary")
            qa_output = gr.Textbox(label="Answer", lines=4, interactive=False)

            qa_btn.click(
                fn=answer_question,
                inputs=[qa_context, qa_question],
                outputs=[qa_output],
            )

            gr.Examples(
                examples=[
                    [
                        "The new Tesla Model 3 has incredible acceleration and the autopilot feature is impressive. However, the interior quality could be better for the price point. The touchscreen interface takes some getting used to.",
                        "What is impressive about the Tesla?",
                    ],
                    [
                        "I've been driving my Toyota Camry for 6 months and it's been extremely reliable. Great fuel economy and comfortable seats for long drives.",
                        "How long has the owner had the car?",
                    ],
                    [
                        "The BMW M3 delivers outstanding performance with its twin-turbo engine. The handling is razor-sharp and the exhaust note is thrilling. Maintenance costs are high but the driving experience is worth every penny.",
                        "What type of engine does the BMW M3 have?",
                    ],
                ],
                inputs=[qa_context, qa_question],
            )

        # ---- Tab 5: Summarization ----
        with gr.Tab("\U0001F4DD Summarization"):
            gr.Markdown("### Summarize lengthy car reviews using T5")

            with gr.Row():
                with gr.Column():
                    summary_input = gr.Textbox(
                        label="Text to Summarize",
                        placeholder="Paste a long car review...",
                        lines=8,
                    )
                    max_len_slider = gr.Slider(
                        minimum=30, maximum=300, value=150, step=10,
                        label="Max Summary Length (tokens)",
                    )
                    summary_btn = gr.Button("Summarize", variant="primary")
                with gr.Column():
                    summary_output = gr.Textbox(
                        label="Summary",
                        lines=8,
                        interactive=False,
                    )

            summary_btn.click(
                fn=summarize_text,
                inputs=[summary_input, max_len_slider],
                outputs=[summary_output],
            )

            gr.Examples(
                examples=[
                    [
                        "The 2024 Toyota Camry continues to be one of the best mid-size sedans on the market. "
                        "It offers a smooth and comfortable ride with excellent fuel economy, achieving up to 39 mpg on the highway. "
                        "The interior is well-designed with quality materials and a user-friendly infotainment system featuring an 8-inch touchscreen. "
                        "Toyota Safety Sense comes standard, providing features like adaptive cruise control, lane departure alert, and automatic emergency braking. "
                        "The 2.5-liter four-cylinder engine produces 203 horsepower and provides adequate power for daily driving. "
                        "The trunk space is generous at 15.1 cubic feet, making it practical for families. "
                        "Overall, the Camry remains a top choice for buyers seeking reliability, efficiency, and comfort in a sedan.",
                        150,
                    ],
                ],
                inputs=[summary_input, max_len_slider],
            )

        # ---- Tab 6: Named Entity Recognition ----
        with gr.Tab("\U0001F50D Named Entity Recognition"):
            gr.Markdown("### Extract named entities (people, organizations, locations) from car reviews")

            with gr.Row():
                with gr.Column():
                    ner_input = gr.Textbox(
                        label="Text for NER",
                        placeholder="Enter text to extract entities...",
                        lines=4,
                    )
                    ner_btn = gr.Button("Extract Entities", variant="primary")
                with gr.Column():
                    ner_output = gr.Textbox(
                        label="Entities Found",
                        lines=8,
                        interactive=False,
                    )

            ner_btn.click(
                fn=extract_entities,
                inputs=[ner_input],
                outputs=[ner_output],
            )

            gr.Examples(
                examples=[
                    ["Tesla CEO Elon Musk announced new features for the Model 3 at the factory in Fremont, California."],
                    ["Toyota and Honda are leading the hybrid market in Japan and the United States."],
                    ["BMW's Munich headquarters revealed the new M3 Competition at the Frankfurt Motor Show."],
                ],
                inputs=[ner_input],
            )

        # ---- Tab 7: Aspect-Based Sentiment ----
        with gr.Tab("\U0001F527 Aspect Analysis"):
            gr.Markdown("### Analyze sentiment for specific automotive aspects (performance, comfort, safety, etc.)")

            with gr.Row():
                with gr.Column():
                    aspect_input = gr.Textbox(
                        label="Car Review",
                        placeholder="Enter a detailed car review...",
                        lines=6,
                    )
                    aspect_btn = gr.Button("Analyze Aspects", variant="primary")
                with gr.Column():
                    aspect_output = gr.Textbox(
                        label="Aspect Sentiments",
                        lines=12,
                        interactive=False,
                    )

            aspect_btn.click(
                fn=analyze_aspect_sentiment,
                inputs=[aspect_input],
                outputs=[aspect_output],
            )

            gr.Examples(
                examples=[
                    [
                        "The engine delivers incredible power and the acceleration is breathtaking. "
                        "The seats are extremely comfortable even on long road trips. "
                        "However, the fuel economy is terrible, barely getting 18 mpg in the city. "
                        "The safety features are top-notch with advanced collision avoidance. "
                        "The infotainment screen is responsive and the navigation works great. "
                        "Build quality feels solid and reliable."
                    ],
                    [
                        "Terrible reliability with constant engine problems. "
                        "The dealer wants a fortune for basic maintenance. "
                        "At least the interior design looks nice and the technology features are modern. "
                        "Fuel efficiency is decent for a car this size."
                    ],
                ],
                inputs=[aspect_input],
            )

        # ---- Sample Data Table ----
        with gr.Accordion("Sample Car Reviews Dataset", open=False):
            gr.Dataframe(
                value=sample_df,
                label="Sample Reviews",
                wrap=True,
            )

        # ---- Footer ----
        gr.HTML("""
        <div class="footer">
            <h3 style="color: #4a90d9; margin-bottom: 15px;">Technical Architecture</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-bottom: 20px;">
                <div class="model-card">
                    <h4>Sentiment Analysis</h4>
                    <p>distilbert-base-uncased-finetuned-sst-2-english</p>
                </div>
                <div class="model-card">
                    <h4>Topic Modeling</h4>
                    <p>BERTopic + all-MiniLM-L6-v2</p>
                </div>
                <div class="model-card">
                    <h4>Translation</h4>
                    <p>Helsinki-NLP/opus-mt-en-es</p>
                </div>
                <div class="model-card">
                    <h4>Question Answering</h4>
                    <p>distilbert-base-cased-distilled-squad</p>
                </div>
                <div class="model-card">
                    <h4>Summarization</h4>
                    <p>t5-small</p>
                </div>
                <div class="model-card">
                    <h4>Named Entity Recognition</h4>
                    <p>dbmdz/bert-large-cased-finetuned-conll03-english</p>
                </div>
                <div class="model-card">
                    <h4>Aspect Analysis</h4>
                    <p>Keyword detection + DistilBERT sentiment</p>
                </div>
            </div>
            <p style="color: #666; font-size: 0.85em;">
                Built with Gradio, Hugging Face Transformers, BERTopic, and Sentence-Transformers<br>
                <a href="https://github.com/Phoenixak99/Analyzing-Car-Reviews-with-LLMs" target="_blank" style="color: #4a90d9;">View Full Project on GitHub</a>
            </p>
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
