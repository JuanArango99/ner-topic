"""End-to-end generation of all poster deliverables:
- Full evaluation metrics: C_V, NPMI, Topic Diversity, Silhouette Score, Outlier %
- Figure 1: Side-by-side UMAP 2D projection (Baseline vs Condition B)
- Figure 2: Grouped bar chart comparing metrics across all conditions
- Table 1: Qualitative comparison of matched clusters
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import spacy.lang.en.stop_words as spacy_stopwords
import torch
from bertopic import BERTopic
from datasets import load_dataset
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import pipeline
from umap import UMAP

# Enable GPU acceleration
spacy.require_gpu()
device_id = 0 if torch.cuda.is_available() else -1
print(f"Device: {'cuda' if device_id == 0 else 'cpu'}")

CACHE_FILE = "data/corpus_extracted_cache.json"

# 1. Dataset Loading & Extraction (with caching)
if os.path.exists(CACHE_FILE):
    print(f"Loading cached aligned corpus from {CACHE_FILE}...")
    with open(CACHE_FILE, "r") as f:
        cached_data = json.load(f)
    paired_raw = cached_data["paired_raw"]
    paired_cond_a = cached_data["paired_cond_a"]
    paired_cond_b = cached_data["paired_cond_b"]
    print(f"Loaded {len(paired_raw)} aligned documents from cache.")
else:
    print("Loading BBC News dataset...")
    dataset = load_dataset("gopalkalpande/bbc-news-summary", split="train")
    raw_corpus = list(dataset["Articles"])

    print("Loading spaCy en_core_web_trf on GPU...")
    nlp = spacy.load("en_core_web_trf", disable=["parser"])
    stopwords = spacy_stopwords.STOP_WORDS

    print("Loading DistilBERT-NER on GPU...")
    ner_pipe = pipeline(
        "ner", model="dslim/distilbert-NER", aggregation_strategy="simple", device=device_id
    )

    # 1A. Batched spaCy extraction
    print("Extracting spaCy entities (GPU batch_size=32)...")
    all_ents_a = []
    for doc in tqdm(nlp.pipe(raw_corpus, batch_size=32), total=len(raw_corpus)):
        clean_tokens = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                for token in ent:
                    tl = token.text.lower()
                    if tl not in stopwords and token.is_alpha and len(tl) > 1:
                        clean_tokens.append(tl)
        all_ents_a.append(clean_tokens)

    # 1B. DistilBERT extraction
    print("Extracting DistilBERT-NER entities...")
    all_ents_b = []
    for text in tqdm(raw_corpus, total=len(raw_corpus)):
        chunks = [p.strip() for p in text.split("\n") if len(p.strip()) > 0]
        if not chunks:
            chunks = [text]
        pipe_results = ner_pipe(chunks)
        if isinstance(pipe_results, dict) or (pipe_results and isinstance(pipe_results[0], dict)):
            pipe_results = [pipe_results]
        clean_tokens = []
        for chunk_ents in pipe_results:
            for ent in chunk_ents:
                if ent["entity_group"] in ["PER", "ORG", "LOC"]:
                    for word in ent["word"].split():
                        wl = word.lower()
                        if wl not in stopwords and wl.isalpha() and len(wl) > 1:
                            clean_tokens.append(wl)
        all_ents_b.append(clean_tokens)

    # Synchronized 1:1:1 alignment
    paired_raw = []
    paired_cond_a = []
    paired_cond_b = []
    for i in range(len(raw_corpus)):
        if all_ents_a[i] and all_ents_b[i]:
            paired_raw.append(raw_corpus[i])
            paired_cond_a.append(" ".join(all_ents_a[i]))
            paired_cond_b.append(" ".join(all_ents_b[i]))

    print(f"Caching {len(paired_raw)} aligned documents to {CACHE_FILE}...")
    with open(CACHE_FILE, "w") as f:
        json.dump(
            {
                "paired_raw": paired_raw,
                "paired_cond_a": paired_cond_a,
                "paired_cond_b": paired_cond_b,
            },
            f,
        )

# 2. Shared Topic Modeling Invariant Setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def build_bertopic_model(seed=42):
    umap_model = UMAP(
        n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=seed
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english")
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=10,
        verbose=False,
    )


print("\n--- Fitting Baseline BERTopic ---")
baseline_model = build_bertopic_model(seed=42)
baseline_topics, _ = baseline_model.fit_transform(paired_raw)
baseline_umap = baseline_model.umap_model.embedding_

print("--- Fitting Condition A (spaCy trf) BERTopic ---")
cond_a_model = build_bertopic_model(seed=42)
cond_a_topics, _ = cond_a_model.fit_transform(paired_cond_a)
cond_a_umap = cond_a_model.umap_model.embedding_

print("--- Fitting Condition B (DistilBERT-NER) BERTopic ---")
cond_b_model = build_bertopic_model(seed=42)
cond_b_topics, _ = cond_b_model.fit_transform(paired_cond_b)
cond_b_umap = cond_b_model.umap_model.embedding_

print("All 3 models fitted successfully!")

# 3. Metrics Evaluation Suite
print("\nCalculating Coherence Models (C_V & NPMI)...")
tokenized_ref = [doc.lower().split() for doc in paired_raw]
dictionary = Dictionary(tokenized_ref)


def evaluate_metrics(model, topics, umap_coords):
    topic_words = []
    all_top_words = []
    topic_dict = model.get_topics()
    for t_id in topic_dict.keys():
        if t_id == -1:
            continue
        words = [w for w, _ in model.get_topic(t_id)[:10]]
        if words:
            topic_words.append(words)
            all_top_words.extend(words)

    # 1. C_V Coherence
    cm_cv = CoherenceModel(
        topics=topic_words, texts=tokenized_ref, dictionary=dictionary, coherence="c_v"
    )
    score_cv = cm_cv.get_coherence()

    # 2. NPMI Coherence
    cm_npmi = CoherenceModel(
        topics=topic_words, texts=tokenized_ref, dictionary=dictionary, coherence="c_npmi"
    )
    score_npmi = cm_npmi.get_coherence()

    # 3. Topic Diversity
    if all_top_words:
        topic_diversity = len(set(all_top_words)) / len(all_top_words)
    else:
        topic_diversity = 0.0

    # 4. Outlier Percentage
    topics_list = list(topics)
    outlier_pct = (topics_list.count(-1) / len(topics_list)) * 100

    # 5. Silhouette Score (on clustered points only)
    topics_arr = np.array(topics)
    mask = topics_arr != -1
    if len(set(topics_arr[mask])) > 1:
        sil_score = float(silhouette_score(umap_coords[mask], topics_arr[mask]))
    else:
        sil_score = 0.0

    num_topics = len(topic_dict) - 1

    return {
        "Topics_Found": num_topics,
        "Outlier_Pct": outlier_pct,
        "C_V": score_cv,
        "NPMI": score_npmi,
        "Topic_Diversity": topic_diversity,
        "Silhouette": sil_score,
    }


metrics_base = evaluate_metrics(baseline_model, baseline_topics, baseline_umap)
metrics_a = evaluate_metrics(cond_a_model, cond_a_topics, cond_a_umap)
metrics_b = evaluate_metrics(cond_b_model, cond_b_topics, cond_b_umap)

df_metrics = pd.DataFrame(
    [
        {"Condition": "Baseline (Raw Text)", "Model": "Raw + Stopwords", **metrics_base},
        {"Condition": "Condition A (spaCy)", "Model": "spaCy (trf)", **metrics_a},
        {"Condition": "Condition B (DistilBERT)", "Model": "DistilBERT-NER", **metrics_b},
    ]
)

df_metrics.to_csv("results/metrics_summary.csv", index=False)
print("\n" + "=" * 80)
print("COMPREHENSIVE METRICS SUMMARY")
print("=" * 80)
print(df_metrics.to_string(index=False))

# 4. Generate Figure 1: 3-Panel UMAP 2D Projection
print("\nGenerating Figure 1: 3-Panel UMAP 2D Projection...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})


def plot_umap(ax, umap_coords, topics, title, num_topics, outlier_pct):
    topics_arr = np.array(topics)
    outliers = topics_arr == -1
    clustered = ~outliers

    # Plot outliers in light gray
    ax.scatter(
        umap_coords[outliers, 0],
        umap_coords[outliers, 1],
        c="#d3d3d3",
        s=12,
        alpha=0.35,
        label=f"Outliers ({outlier_pct:.1f}%)",
    )

    # Plot clusters with categorical colormap
    ax.scatter(
        umap_coords[clustered, 0],
        umap_coords[clustered, 1],
        c=topics_arr[clustered],
        cmap="tab20",
        s=16,
        alpha=0.85,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)


plot_umap(
    ax1,
    baseline_umap,
    baseline_topics,
    f"Baseline: Raw Text\n({metrics_base['Topics_Found']} topics, {metrics_base['Outlier_Pct']:.1f}% outliers)",
    metrics_base["Topics_Found"],
    metrics_base["Outlier_Pct"],
)

plot_umap(
    ax2,
    cond_a_umap,
    cond_a_topics,
    f"Condition A: spaCy (trf)\n({metrics_a['Topics_Found']} topics, {metrics_a['Outlier_Pct']:.1f}% outliers)",
    metrics_a["Topics_Found"],
    metrics_a["Outlier_Pct"],
)

plot_umap(
    ax3,
    cond_b_umap,
    cond_b_topics,
    f"Condition B: DistilBERT-NER\n({metrics_b['Topics_Found']} topics, {metrics_b['Outlier_Pct']:.1f}% outliers)",
    metrics_b["Topics_Found"],
    metrics_b["Outlier_Pct"],
)

fig.suptitle(
    "Figure 1: Side-by-Side UMAP 2D Embedding Space Projections Across Conditions",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
fig.tight_layout()
fig.savefig("figures/figure1_umap_comparison.png", dpi=300)
print("Saved Figure 1 to figure1_umap_comparison.png")

# 5. Generate Figure 2: Grouped Bar Chart of Metrics
print("\nGenerating Figure 2: Grouped Bar Chart...")
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

conditions = ["Baseline\n(Raw Text)", "Condition A\n(spaCy trf)", "Condition B\n(DistilBERT-NER)"]
x = np.arange(len(conditions))
width = 0.20

cv_scores = [metrics_base["C_V"], metrics_a["C_V"], metrics_b["C_V"]]
npmi_scores = [metrics_base["NPMI"], metrics_a["NPMI"], metrics_b["NPMI"]]
div_scores = [
    metrics_base["Topic_Diversity"],
    metrics_a["Topic_Diversity"],
    metrics_b["Topic_Diversity"],
]
sil_scores = [metrics_base["Silhouette"], metrics_a["Silhouette"], metrics_b["Silhouette"]]

r1 = ax.bar(
    x - 1.5 * width,
    cv_scores,
    width,
    label="C_V Coherence [0, 1]",
    color="#1f77b4",
    edgecolor="black",
    alpha=0.9,
)
r2 = ax.bar(
    x - 0.5 * width,
    npmi_scores,
    width,
    label="NPMI Coherence [-1, 1]",
    color="#ff7f0e",
    edgecolor="black",
    alpha=0.9,
)
r3 = ax.bar(
    x + 0.5 * width,
    div_scores,
    width,
    label="Topic Diversity [0, 1]",
    color="#2ca02c",
    edgecolor="black",
    alpha=0.9,
)
r4 = ax.bar(
    x + 1.5 * width,
    sil_scores,
    width,
    label="Silhouette Score [-1, 1]",
    color="#9467bd",
    edgecolor="black",
    alpha=0.9,
)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        val_str = f"{height:.2f}"
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            val_str,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )


autolabel(r1)
autolabel(r2)
autolabel(r3)
autolabel(r4)

ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title(
    "Figure 2: Multi-Metric Comparison Across Conditions", fontsize=13, fontweight="bold", pad=15
)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylim(-0.15, 0.95)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

fig.tight_layout()
fig.savefig("figures/figure2_metrics_barchart.png", dpi=300)
print("Saved Figure 2 to figure2_metrics_barchart.png")

# 6. Generate Table 1: Qualitative Top-Words Comparison
print("\nGenerating Table 1: Qualitative Comparison...")


def find_topic_by_keywords(model, keywords):
    best_t = -1
    best_match = 0
    for t_id in model.get_topics().keys():
        if t_id == -1:
            continue
        top_words = [w for w, _ in model.get_topic(t_id)[:15]]
        matches = sum(1 for kw in keywords if kw.lower() in top_words)
        if matches > best_match:
            best_match = matches
            best_t = t_id
    if best_t != -1:
        words = [w for w, _ in model.get_topic(best_t)[:8]]
        count = len([t for t in model.topics_ if t == best_t])
        return f"{', '.join(words)} (n={count})"
    return "N/A"


thematic_queries = [
    ("UK Politics & Government", ["labour", "blair", "brown", "election", "party", "minister"]),
    (
        "Football / Premier League",
        ["chelsea", "arsenal", "united", "league", "liverpool", "ferguson", "cup"],
    ),
    ("Tennis / Grand Slam", ["roddick", "federer", "nadal", "hewitt", "seed", "open", "tennis"]),
    (
        "Cinema / Academy Awards",
        ["film", "actor", "oscars", "awards", "hollywood", "actress", "director"],
    ),
    (
        "Business & Stock Exchanges",
        ["shares", "deutsche", "boerse", "firm", "company", "market", "lse"],
    ),
]

table1_rows = []
for theme, kws in thematic_queries:
    base_res = find_topic_by_keywords(baseline_model, kws)
    cond_a_res = find_topic_by_keywords(cond_a_model, kws)
    cond_b_res = find_topic_by_keywords(cond_b_model, kws)
    table1_rows.append(
        {
            "Thematic Domain": theme,
            "Baseline (Raw Text)": base_res,
            "Condition A (spaCy trf)": cond_a_res,
            "Condition B (DistilBERT-NER)": cond_b_res,
        }
    )

df_table1 = pd.DataFrame(table1_rows)
df_table1.to_csv("results/table1_qualitative_comparison.csv", index=False)
print("\nTABLE 1: QUALITATIVE COMPARISON")
print(df_table1.to_string(index=False))

print("\nAll deliverables generated successfully!")
