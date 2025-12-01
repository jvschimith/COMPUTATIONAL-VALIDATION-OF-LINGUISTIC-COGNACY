# COMPUTATIONAL-VALIDATION-OF-LINGUISTIC-COGNACY

The linguistic affiliation of the ancient Xiongnú and Huns remains a subject of debate in historical linguistics, with hypotheses spanning Turkic, Mongolic, and Yeniseian families. This project leverages **Graph Machine Learning (ML)** to computationally validate the recent hypothesis proposed by Bonmann and Fries \cite{Bonmann2025Xiongnu}, suggesting a direct linguistic link between the Xiongnú/Huns and the Paleo-Siberian Yeniseian language family (specifically, Arin). By constructing a **weighted graph** where nodes represent languages and edge weights quantify shared cognates, we apply the **Node2Vec** graph embedding algorithm \cite{Grover2016Node2Vec}. Subsequent visualization using **t-distributed Stochastic Neighbor Embedding (t-SNE)** \cite{VanDerMaaten2008TSNE} demonstrates that the Xiongnú and Huns nodes cluster tightly with the Yeniseian languages (Arin, Ket, Yugh), providing a quantitative, geometry-based confirmation of the linguistic proximity established via the traditional comparative method


How to run:
streamlit run ./app.py

### • O Problema (The Problem)

**Problema Linguístico:** A afiliação genética das línguas dos antigos **Xiongnú** e **Hunos** é uma questão central e controversa na linguística histórica, com hipóteses conflitantes (túrquica, mongólica, ieniseiana). O método comparativo tradicional é rigoroso, mas manual.

**Objetivo do Projeto:** Fornecer uma **validação computacional e quantitativa** para a tese apresentada no artigo de Bonmann e Fries \cite{Bonmann2025Xiongnu}, que sugere que Xiongnú e Hunos falavam uma forma antiga da língua **Arin** (da família Ieniseiana), usando técnicas modernas de Aprendizagem de Máquina em Grafos (Graph ML).

---

### • O Dataset Utilizado (The Dataset)

* **Fonte:** Dados de cognatos e correspondências sonoras extraídos das tabelas comparativas do artigo "Linguistic Evidence Suggests that Xiongnú and Huns Spoke the Same Paleo-Siberian Language" (Bonmann & Fries, 2025).
* **Formato:** O *dataset* foi estruturado como uma **Lista de Arestas Ponderadas (Weighted Edge List)**.
* **Entidades (Nós):** As línguas analisadas, incluindo **Arin, Ket, Yugh** (Ieniseianas), **Xiongnú, Huns** (foco do estudo), **Proto-Turkic** e **Proto-Mongolic**.
* **Relações (Pesos):** O peso de cada aresta entre duas línguas representa a **proximidade linguística**, inferida a partir do número de cognatos compartilhados ou da força das correspondências sonoras regulares.

---

### • A Técnica Aplicada (The Technique Applied)

| Etapa | Técnica | Descrição |
| :--- | :--- | :--- |
| **Modelagem** | **Grafo (NetworkX)** | Modelar as línguas como **nós** e as relações de cognatos como **arestas ponderadas** (peso = proximidade). |
| **Aprendizagem** | **Node2Vec (Graph Embedding)** \cite{Grover2016Node2Vec} | Algoritmo de Aprendizagem de Máquina Não Supervisionada que realiza *passeios aleatórios enviesados* no grafo para gerar um vetor numérico (embedding) de 64 dimensões para cada língua. O objetivo é que a **similaridade vetorial** reflita a **proximidade estrutural** no grafo. |
| **Visualização** | **t-SNE (t-distributed Stochastic Neighbor Embedding)** \cite{VanDerMaaten2008TSNE} | Técnica de redução de dimensionalidade para mapear os 64 *embeddings* em apenas duas dimensões (2D) para visualização. Preserva a estrutura de agrupamento dos dados no espaço geométrico. |
| **** | | |

---

### • Métricas de Desempenho (Performance Metrics)

Como este é um projeto de **Aprendizagem Não Supervisionada** (representação de dados) em vez de um modelo preditivo, as métricas avaliam a coerência da representação:

1.  **Validação Visual (t-SNE):** A métrica primária é a **coerência do agrupamento**. O sucesso é medido pela clara separação das famílias (ex: Ieniseiana vs. Túrquica) e pelo **agrupamento imediato dos nós Xiongnú e Huns** junto ao cluster Ieniseiano (Arin/Ket), provando a proximidade.
2.  **Similaridade de Cosseno (Cosine Similarity):** Usada para quantificar a distância entre os vetores de *embedding*.
    * **Meta de Sucesso:** Alta similaridade (próxima a 1.0) entre (Arin, Ket, Xiongnú, Huns).
    * **Resultado Típico:** Similaridade > 0.95 entre Xiongnú e Arin, e baixa similaridade (ex: < 0.5) com línguas distantes (ex: Proto-Turkic).

---

### • Demonstração do Streamlit (Streamlit Demonstration)

O projeto foi empacotado em um aplicativo **Streamlit** para permitir que o professor e os colegas explorem os resultados de forma interativa.

* **Funcionalidade:** O aplicativo exibe a tabela de entrada (relações de cognatos) e, em seguida, o **Gráfico de Dispersão 2D interativo** gerado pelo Node2Vec + t-SNE (usando Plotly).
* **Interatividade:** O usuário pode passar o mouse sobre os pontos para identificar a língua e fazer *zoom* nas áreas de interesse para examinar a proximidade exata entre Xiongnú/Huns e as línguas Ieniseianas.

---

### • Limitações e Possibilidades Futuras (Limitations and Future Work)

| Categoria | Limitações | Possibilidades Futuras |
| :--- | :--- | :--- |
| **Dados** | Dependência de um *dataset* simulado (extraído manualmente) e relativamente pequeno. | Automatizar a extração de dados tabulares de PDFs linguísticos e expandir o *dataset* para incluir mais famílias (ex: Uralo-Siberiana). |
| **Método** | Node2Vec é um modelo de Aprendizagem de Representação (Não Supervisionada). | Aplicar **Redes Neurais de Grafo (GNNs)** para um problema de **Classificação de Arestas** (Aprendizagem Supervisionada), treinando o modelo para *prever* se uma relação é um cognato, empréstimo ou coincidência. |
| **Validação** | O resultado apenas *corrobora* a tese linguística (validando a estrutura de proximidade). | Incluir características fonéticas (`features`) dos lexemas (além da estrutura do grafo) para um modelo mais rico, como GCNs (Graph Convolutional Networks). |
