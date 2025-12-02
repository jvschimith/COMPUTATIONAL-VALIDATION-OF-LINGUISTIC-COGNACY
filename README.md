# **COMPUTATIONAL VALIDATION OF LINGUISTIC COGNACY**

*Luiz Otávio e João Victor*

The linguistic affiliation of the ancient Xiongnú and Huns remains a subject of debate in historical linguistics, with hypotheses spanning the Turkic, Mongolic, and Yeniseian families. This project leverages **Graph Machine Learning (ML)** to computationally validate the recent hypothesis proposed by Bonmann and Fries {Bonmann2025Xiongnu}, suggesting a direct linguistic link between the Xiongnú/Huns and the Paleo-Siberian Yeniseian language family (specifically Arin). By constructing a **weighted graph** where nodes represent languages and edge weights quantify shared cognates, we apply the **Node2Vec** graph embedding algorithm {Grover2016Node2Vec}. Subsequent visualization using **t-distributed Stochastic Neighbor Embedding (t-SNE)** {VanDerMaaten2008TSNE} shows that the Xiongnú and Huns nodes cluster tightly with the Yeniseian languages (Arin, Ket, Yugh), providing a quantitative, geometry-based confirmation of the linguistic proximity established by the traditional comparative method.

**How to run:**

```
streamlit run ./app.py
```

---

## • The Problem

**Linguistic Problem:**
The genetic affiliation of the languages of the ancient **Xiongnú** and **Huns** is a central and controversial question in historical linguistics, with conflicting hypotheses (Turkic, Mongolic, Yeniseian). The traditional comparative method is rigorous but manual.

**Project Goal:**
To provide a **computational and quantitative validation** for the thesis proposed in the article by Bonmann and Fries \cite{Bonmann2025Xiongnu}, which suggests that Xiongnú and Huns spoke an ancient form of the **Arin** language (from the Yeniseian family), using modern Graph Machine Learning (Graph ML) techniques.

---

## • The Dataset Used

* **Source:** Cognate and sound-correspondence data extracted from the comparative tables in *“Linguistic Evidence Suggests that Xiongnú and Huns Spoke the Same Paleo-Siberian Language”* (Bonmann & Fries, 2025).
* **Format:** The dataset was structured as a **Weighted Edge List**.
* **Entities (Nodes):** The analyzed languages, including **Arin, Ket, Yugh** (Yeniseian), **Xiongnú, Huns** (focus of the study), **Proto-Turkic**, and **Proto-Mongolic**.
* **Relations (Weights):** The weight of each edge between two languages represents **linguistic proximity**, inferred from the number of shared cognates or the strength of regular sound correspondences.

---

## • The Technique Applied

| Step              | Technique                                                | Description                                                                                                                                                                                                                                            |
| :---------------- | :------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Modeling**      | **Graph (NetworkX)**                                     | Languages are modeled as **nodes**, and cognate relations as **weighted edges** (weight = proximity).                                                                                                                                                  |
| **Learning**      | **Node2Vec (Graph Embedding)** \cite{Grover2016Node2Vec} | An unsupervised ML algorithm that performs **biased random walks** on the graph to generate a 64-dimensional numerical vector (*embedding*) for each language. The goal is for **vector similarity** to reflect **structural proximity** in the graph. |
| **Visualization** | **t-SNE** \cite{VanDerMaaten2008TSNE}                    | Dimensionality-reduction technique mapping the 64-dimensional embeddings into a 2D plane. Preserves cluster structure for visual interpretation.                                                                                                       |
|                   |                                                          |                                                                                                                                                                                                                                                        |

---

## • Performance Metrics

Because this is an **Unsupervised Learning** project (data representation) rather than a predictive model, the metrics evaluate representational coherence:

1. **Visual Validation (t-SNE):**
   The primary metric is **cluster coherence**. Success is measured by clear separation between families (e.g., Yeniseian vs. Turkic) and the **tight clustering of Xiongnú and Huns** within the Yeniseian group (Arin/Ket), demonstrating proximity.

2. **Cosine Similarity:**
   Used to quantify the distance between embedding vectors.

   * **Success Goal:** High similarity (close to 1.0) among (Arin, Ket, Xiongnú, Huns).
   * **Typical Result:** Similarity > 0.95 between Xiongnú and Arin, and low similarity (e.g., < 0.5) with distant languages (e.g., Proto-Turkic).

---

## • Streamlit Demonstration

The project is deployed as a **Streamlit** application for interactive exploration by the professor and colleagues.

* **Functionality:**
  The app displays the input table (cognate relations) and, afterward, the **interactive 2D scatter-plot** generated by Node2Vec + t-SNE (using Plotly).

* **Interactivity:**
  Users can hover over points to see language labels and zoom into clusters to inspect the exact proximity between Xiongnú/Huns and Yeniseian languages.

---

## • Limitations and Future Work

| Category       | Limitations                                                                               | Future Work                                                                                                                                                                               |
| :------------- | :---------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data**       | Dependence on a manually extracted dataset, relatively small in size.                     | Automate extraction of tabular data from linguistic PDFs and expand to more families (e.g., Uralo-Siberian).                                                                              |
| **Method**     | Node2Vec is an unsupervised representation-learning model.                                | Apply **Graph Neural Networks (GNNs)** to an **edge-classification** problem (supervised), training the model to *predict* whether a relation is a cognate, a loanword, or a coincidence. |
| **Validation** | The result only *corroborates* the linguistic hypothesis (validates proximity structure). | Add phonetic features of lexemes (beyond graph structure) to create richer models, such as GCNs (Graph Convolutional Networks).                                                           |

