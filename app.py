import streamlit as st
import pandas as pd
import networkx as nx
import editdistance 
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import plotly.express as px
from itertools import combinations
import warnings

# --- CONFIGURAÇÃO INICIAL STREAMLIT ---
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore") # Ignora warnings

# ----------------------------------------------------------------------
# A. DADOS INICIAIS (Defaults para o Editor de Dados)
# ----------------------------------------------------------------------

# Mapeamento padrão de Famílias para inicializar o sidebar
DEFAULT_FAMILY_MAPPING = {
    'Arin': 'Yeniseiana', 
    'Xiongnú': 'Xiongnú/Huns (Foco)', 
    'Proto-Turkic': 'Controle (Turkic)', 
    'Huns': 'Xiongnú/Huns (Foco)', 
    'Ket': 'Yeniseiana',
    'Proto-Mongolic': 'Controle (Mongolic)'
}

# Dados de palavras padrão para inicializar o editor de dados
DEFAULT_WORDS_DATA = {
    'Conceito': ['Água', 'Dois', 'Pássaro', 'Fogo', 'Dedo', 'Comer', 'Ovo', 'Mãe', 'Nariz', 'Dente'],
    'Arin': ['yit', 'kin', 'qun', 'si', 'tū', 'ēsi', 'uśe', 'ami', 'qan', 'qan'],
    'Xiongnú': ['yyt', 'k’in', 'qun', 'sa', 't\'u', 'ēssi', 'use', 'amy', 'qani', 'qan'],
    'Proto-Turkic': ['su', 'eki', 'quş', 'ot', 'til', 'ye', 'yumu', 'ana', 'burun', 'tiš'],
    'Huns': ['yit', 'kin', 'cun', 'se', 'tu', 'esi', 'use', 'amy', 'qan', 'qann'],
    'Ket': ['u’l', 'qīn', 'qun', 'sī', 'dū', 'e’s', 'ūs', 'amī', 'qan', 'qa'],
    'Proto-Mongolic': ['usu', 'qoyar', 'šuγ', 'γal', 'urγu', 'ide', 'öndü', 'eke', 'qabar', 'sidü']
}

# ----------------------------------------------------------------------
# B. FUNÇÕES DE PROCESSAMENTO
# ----------------------------------------------------------------------

@st.cache_data
def calculate_pair_similarity(lang1_series, lang2_series):
    """Calcula a similaridade média normalizada por edit distance entre duas listas de palavras."""
    total_similarity_score = 0
    num_concepts = len(lang1_series)
    
    for word1, word2 in zip(lang1_series, lang2_series):
        dist = editdistance.eval(word1, word2)
        max_len = max(len(word1), len(word2), 1) 
        normalized_similarity = 1 - (dist / max_len)
        total_similarity_score += normalized_similarity
        
    avg_similarity = total_similarity_score / num_concepts
    # Multiplica por 20 para ter pesos de aresta mais visíveis (escalonamento)
    final_weight = avg_similarity * 20 
    
    return final_weight


@st.cache_data
def generate_weighted_edges(df, languages):
    """Gera o DataFrame de arestas ponderadas (input para o Node2Vec)."""
    weighted_edges = []
    
    for lang1, lang2 in combinations(languages, 2):
        # Acessa as colunas de palavras no DataFrame de input
        weight = calculate_pair_similarity(df[lang1], df[lang2])
        weighted_edges.append({
            'Source': lang1,
            'Target': lang2,
            'Weight': round(weight, 2)
        })
        
    return pd.DataFrame(weighted_edges)


@st.cache_resource
def run_node2vec_analysis(input_df, family_mapping):
    """Executa a criação do grafo, Node2Vec e t-SNE."""

    # 1. Grafo
    G = nx.Graph()
    for index, row in input_df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    # 2. Node2Vec 
    node2vec = Node2Vec(
        G,
        walk_length=20,
        num_walks=200,
        p=1, 
        q=1,
        weight_key='weight',
        workers=4 # Define o número de threads
    )

    # 3. Word2Vec com API NOVA (vector_size) - Corrige o TypeError
    model = node2vec.fit(
        vector_size=64,   # Parâmetro correto para Gensim 4.x
        window=10,
        min_count=1,
        sg=1,             # skip-gram
        batch_words=32,
        epochs=20
    )

    # 4. Extrai embeddings
    embeddings = {node: model.wv[node] for node in G.nodes()}
    embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embedding_df.index.name = 'Língua'

    # 5. t-SNE
    X = embedding_df.values
    language_labels = embedding_df.index.tolist()

    # Perplexity precisa ser menor que (N - 1)
    perplexity_val = min(5, len(G.nodes()) - 1)
    if perplexity_val <= 0:
        return pd.DataFrame()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity_val,
        random_state=42,
        n_iter=5000
    )

    X_tsne = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(
        X_tsne,
        columns=['Componente 1 (t-SNE)', 'Componente 2 (t-SNE)'],
        index=language_labels
    )
    tsne_df['Língua'] = tsne_df.index

    # Classificação das famílias com base no input do usuário
    def get_family(lang):
        return family_mapping.get(lang, 'Família Desconhecida')

    tsne_df['Família Linguística'] = tsne_df['Língua'].apply(get_family)

    return tsne_df

# ----------------------------------------------------------------------
# C. INTERFACE DE USUÁRIO (Input)
# ----------------------------------------------------------------------

st.title("Validação da Hipótese Linguística via Node2Vec (Input Dinâmico) 📊")
st.markdown("""
Use a barra lateral para configurar o mapeamento de línguas e edite a tabela abaixo para inserir seus dados de lexemas.
""")

# --- 1. Sidebar Input: Mapeamento Linguagem-Família ---
with st.sidebar:
    st.header("⚙️ Configuração de Dados")
    st.subheader("1. Mapeamento Linguagem - Família")
    st.markdown("Defina a qual família cada língua pertence. Adicione ou remova linhas conforme necessário.")
    
    family_df_default = pd.DataFrame(
        list(DEFAULT_FAMILY_MAPPING.items()), 
        columns=['Língua', 'Família Linguística']
    )
    
    family_df_input = st.data_editor(
        family_df_default,
        key="family_map_editor",
        num_rows="dynamic",
        use_container_width=True
    )

# Processa o mapeamento
if not family_df_input.empty:
    family_map = family_df_input.set_index('Língua')['Família Linguística'].to_dict()
else:
    st.error("O mapeamento de Linguagem para Família não pode estar vazio.")
    st.stop()
    
LANGUAGES = list(family_map.keys())

# --- 2. Main Input: Lexemas Fonéticos ---
st.header("1. Análise de Similaridade de Palavras (Feature Engineering)")
st.markdown("---")

st.subheader("1.1 Entrada de Dados: Lexemas Fonéticos (Editável)")
st.markdown(f"""
Edite a tabela. As colunas devem incluir o `Conceito` e as **{len(LANGUAGES)}** línguas definidas: {', '.join(LANGUAGES)}.
""")

words_df_input = st.data_editor(
    pd.DataFrame(DEFAULT_WORDS_DATA),
    key="words_data_editor",
    num_rows="dynamic",
    use_container_width=True,
)

# Verifica a integridade dos dados antes de prosseguir
required_columns = set(LANGUAGES)
available_columns = set(words_df_input.columns)

if not required_columns.issubset(available_columns):
    missing_cols = required_columns - available_columns
    st.error(f"Erro: A tabela de Lexemas está faltando as seguintes colunas de Línguas definidas na sidebar: {', '.join(missing_cols)}")
    st.stop()

# ----------------------------------------------------------------------
# D. EXECUÇÃO DA ANÁLISE E OUTPUT
# ----------------------------------------------------------------------

# 1. Geração dos Pesos (Arestas Ponderadas)
try:
    languages_df = generate_weighted_edges(words_df_input, LANGUAGES)
except KeyError as e:
    st.error(f"Erro na geração de pesos. Verifique se as colunas das línguas na tabela de lexemas correspondem exatamente às línguas definidas na barra lateral. Detalhe do erro: {e}")
    st.stop()

st.subheader("1.2 Output: Pesos Calculados (Arestas Ponderadas)")
st.markdown("O valor de `Weight` (Peso) é a pontuação de proximidade de similaridade fonética e é o **input para o Node2Vec**.")
st.dataframe(languages_df, use_container_width=True)


st.header("2. Pipeline de Machine Learning e Prova Geométrica")
st.markdown("---")

# 2. Executa a Análise Node2Vec + t-SNE
tsne_results = run_node2vec_analysis(languages_df, family_map)

st.subheader("2.1 Visualização: Agrupamento Node2Vec + t-SNE")
st.markdown("""

O grafo de dispersão mostra as línguas mapeadas em 2D. A proximidade física reflete a **alta Similaridade de Cosseno** entre os *embeddings* de 64 dimensões.
""")

if not tsne_results.empty:
    fig = px.scatter(
        tsne_results,
        x='Componente 1 (t-SNE)',
        y='Componente 2 (t-SNE)',
        color='Família Linguística',
        text='Língua',
        title="Agrupamento de Línguas (Node2Vec Embeddings)",
        height=600,
        hover_data=['Língua', 'Família Linguística']
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Não foi possível gerar resultados t-SNE. O número de línguas pode ser insuficiente.")

st.subheader("2.2 Conclusão do Modelo")
st.markdown("""
A análise fornece uma **prova geométrica computacional** da hipótese ao observar o agrupamento das línguas no espaço 2D. 
O resultado é dinâmico e depende dos seus dados de entrada (lexemas) e do mapeamento de famílias.
""")
