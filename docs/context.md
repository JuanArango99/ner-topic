# Research Specification: Entity-Centric Topic Modeling with BERTopic

## 1. Project Overview & Meta
* **Title:** Does NER Filtering Improve Topic Modeling? A comparative study of NER-filtered vs raw text input for BERTopic clustering[cite: 2].
* **Context:** Master's-level NLP research seminar, University of Trier.
* **Target Deliverable:** DIN A1 Academic Poster (Submission Deadline: September 30).
* **Hardware Profile:** Local execution on Debian Linux, NVIDIA RTX 3050 (4GB VRAM), 16GB System RAM. All models must run locally within this memory budget[cite: 2].

---

## 2. Research Questions & Hypotheses
* **RQ1 (Primary / Falsifiable):** Does NER-filtered input produce statistically more coherent BERTopic clusters than raw text[cite: 2]?
  * **H1 (Main):** NER-filtered input will produce higher $C_V$ coherence scores than raw text input, because entity tokens carry higher topical signal density than arbitrary vocabulary[cite: 2].
* **RQ2 (Model Quality):** Does NER model quality (`spaCy` vs `DistilBERT-NER`) affect downstream topic coherence[cite: 2]?
  * **H2:** `DistilBERT-NER` will produce higher coherence than `spaCy` due to higher NER precision on the CoNLL-2003 benchmark[cite: 2].
* **RQ3 (Causal Mechanism / Novel Contribution):** Does NER precision act as a causal mechanism — do NER errors degrade topic coherence proportionally[cite: 2]?
  * **H3 (Error Propagation):** Topic coherence will increase monotonically as the NER confidence threshold rises ($\tau \in [0.30, 0.95]$), confirming that NER false positives directly introduce topic noise[cite: 2].

---

## 3. Experimental Matrix
1. **Baseline (Control):** Raw preprocessed text $\rightarrow$ BERTopic[cite: 2]. Uses `CountVectorizer(stop_words="english")` at the c-TF-IDF stage so the baseline is not penalized by functional English grammar.
2. **Condition A:** `spaCy` (`en_core_web_trf`) $\rightarrow$ Extracts `PER`, `ORG`, `LOC`/`GPE` $\rightarrow$ BERTopic[cite: 2].
3. **Condition B:** `DistilBERT-NER` (`dslim/distilbert-NER` fine-tuned on CoNLL-2003) $\rightarrow$ Extracts `PER`, `ORG`, `LOC` $\rightarrow$ BERTopic[cite: 2].
4. **Idea 1B (Threshold Sweep):** Parametric sweep over spaCy entity confidence thresholds $\tau \in [0.30, 0.50, 0.70, 0.85, 0.95]$[cite: 2]. Refilter corpus at each $\tau$, retrain BERTopic, and plot Coherence vs Precision to produce the **Error Propagation Curve**[cite: 2].

---

## 4. Models & Fixed Hyperparameters
To isolate the effect of entity filtering, all downstream topic modeling parameters remain strictly invariant across conditions[cite: 2]:
* **Document Embedder:** `SentenceTransformers("all-MiniLM-L6-v2")` (384-dimensional dense vectors; fixed random seed)[cite: 2].
* **Dimensionality Reduction:** UMAP (reduces embeddings to 2D for clustering and visualization)[cite: 2].
* **Density Clustering:** HDBSCAN with `min_cluster_size=20` fixed across all conditions (yields ~15–40 topics)[cite: 2].
* **Topic Representation:** BERTopic class-based TF-IDF (`c-TF-IDF`)[cite: 2].
* **NER Models:**
  * `en_core_web_trf`: spaCy transformer pipeline; GPU-accelerated; exposes per-entity confidence scores for Idea 1B[cite: 2].
  * `dslim/distilbert-NER`: HuggingFace token classifier; fixed filter[cite: 2].

---

## 5. Dataset Strategy
* **Target Corpus Options:**
  1. **UN General Assembly Debates (`un-general-debates.csv`):** Formal diplomatic transcripts, high entity density (`GPE`, `ORG`), zero HTML artifacts. Highly suited for geopolitical topic clustering.
  2. **BBC News (`SetFit/bbc-news`):** 2,225 full-length journalistic articles across 5 domains (Business, Entertainment, Politics, Sport, Tech).
* **Corpus Preprocessing Rule:** Zero-entity documents must be removed in synchronized pairs across baseline and condition datasets to maintain a strict 1:1 evaluation index. Internal stopwords inside entity spans must be stripped.

---

## 6. Evaluation Framework
* **$C_V$ Coherence (Primary Quantitative):** Evaluates semantic co-occurrence of top-10 topic words using a sliding window against the **raw, unfiltered reference corpus** via Gensim's `CoherenceModel`[cite: 2]. Range: $[0, 1]$[cite: 2].
* **NPMI (Secondary Quantitative):** Normalized Pointwise Mutual Information (range $[-1, +1]$) as a corroborating co-occurrence metric[cite: 2].
* **Topic Diversity:** Percentage of unique words across all generated topics to detect vocabulary starvation or topic collapse[cite: 2].
* **Silhouette Score:** Evaluates geometric separation and cluster tightness in the 2D UMAP embedding space[cite: 2].
* **Qualitative Readability:** Side-by-side inspection of top-10 terms for matched clusters across Baseline, Condition A, and Condition B[cite: 2].

---

## 7. Expected Deliverables (DIN A1 Poster)
* **Figure 1:** Side-by-side UMAP 2D projection (Baseline vs Condition B) showing cluster separation[cite: 2].
* **Figure 2:** Bar chart of $C_V$ and NPMI scores comparing Baseline, Condition A, and Condition B[cite: 2].
* **Figure 3:** Line chart displaying the Error Propagation Curve (Coherence vs NER Confidence Threshold $\tau$)[cite: 2].
* **Table 1:** Qualitative comparison table showing top-10 words for the same thematic cluster across all conditions[cite: 2].

---

## 8. Theoretical Anchor References
* **BERTopic:** Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794[cite: 2].
* **Topic Coherence ($C_V$):** Röder, M., Both, A., & Hinneburg, A. (2015). *Exploring the space of topic coherence measures.* WSDM[cite: 2].
* **NER Foundations:** Tjong Kim Sang, E., & De Meulder, F. (2003). *Introduction to the CoNLL-2003 shared task.* ACL[cite: 2].
* **Sentence-BERT:** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-Networks.* EMNLP[cite: 2].
* **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection.* arXiv:1802.03426[cite: 2].