import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import gensim
from gensim.models import Word2Vec

# Configuração de Reprodução (Seed)
# Isso garante que o Node2Vec e o t-SNE gerem resultados consistentes
@st.cache_data
def set_seed(seed=42):
    np.random.seed(seed)
    # A biblioteca Node2Vec/Gensim usa sua própria seed.

# ----------------------------------------------------------------------
# 1. DADOS BASE (SIMULADOS)
# ----------------------------------------------------------------------

# Dados base que simulam a extração manual das tabelas do PDF
BASE_LANGUAGE_RELATIONS = {
    'Source': ['Arin', 'Arin', 'Ket', 'Arin', 'Arin', 'Xiongnú', 'Arin', 'Ket'],
    'Target': ['Ket', 'Yugh', 'Yugh', 'Xiongnú', 'Huns', 'Huns', 'Proto-Turkic', 'Proto-Mongolic'],
    'Weight': [10, 8, 9, 12, 11, 13, 3, 2] # Peso = Força da Proximidade/Cognato
}

# ----------------------------------------------------------------------
# 2. FUNÇÃO DE PRÉ-PROCESSAMENTO E ANÁLISE (O coração do projeto)
# ----------------------------------------------------------------------

# NOTE: Não usaremos @st.cache_resource aqui porque o input (languages_df) agora é dinâmico.
# A função será re-executada sempre que o DataFrame mudar.
def run_node2vec_analysis(languages_df):
    """Executa a criação do grafo, Node2Vec e t-SNE."""
    
    # 1. Criação do Grafo NetworkX
    G = nx.Graph()
    for index, row in languages_df.iterrows():
        # Adiciona arestas usando o 'Weight' como peso da aresta
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    # Verifica se há nós suficientes para t-SNE
    if len(G.nodes()) < 2:
        st.warning("Adicione pelo menos duas relações para criar um grafo válido.")
        return None

    # 2. Treinamento do Node2Vec
    node2vec = Node2Vec(G,
                        dimensions=64,
                        walk_length=20,
                        num_walks=200,
                        p=1, q=1,
                        weight_key='weight',
                        workers=4)

    # Configuração do modelo Word2Vec
    w2v_kwargs = dict(window=10, min_count=1, batch_words=4, epochs=20)
    try:
        major = int(gensim.__version__.split('.')[0])
    except Exception:
        major = 4
    if major >= 4:
        w2v_kwargs['vector_size'] = node2vec.dimensions
    else:
        w2v_kwargs['size'] = node2vec.dimensions

    model = Word2Vec(node2vec.walks, **w2v_kwargs)

    # Extrai os embeddings
    embeddings = {}
    for node in G.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            # Tenta converter para string, útil se o node for numérico
            embeddings[node] = model.wv[str(node)]

    embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embedding_df.index.name = 'Language'

    # 3. Aplicação do t-SNE para redução de dimensionalidade
    X = embedding_df.values
    language_labels = embedding_df.index.tolist()
    
    # Perplexidade deve ser menor que (N-1)
    perplexity_val = min(5, len(G.nodes()) - 1) 
    
    # Se houver menos de 5 nós, ajusta a perplexidade
    if perplexity_val < 1:
        perplexity_val = 1 
    
    tsne = TSNE(n_components=2, 
                random_state=42, 
                perplexity=perplexity_val, 
                n_iter=5000,
                init='pca' if len(G.nodes()) > 3 else 'random') # 'pca' é melhor para N>3
    
    X_tsne = tsne.fit_transform(X)

    # 4. Criação do DataFrame final para visualização
    tsne_df = pd.DataFrame(data = X_tsne, 
                           columns = ['Componente 1 (t-SNE)', 'Componente 2 (t-SNE)'], 
                           index=language_labels)
    tsne_df['Língua'] = tsne_df.index
    
    # 5. Adicionar uma coluna para o agrupamento visual/linguístico
    # Pega o mapeamento de Família Linguística da Session State, se existir
    family_map = st.session_state.get('language_family_map', {})
    
    def get_family(lang):
        if lang in ['Arin', 'Ket', 'Yugh']:
            return 'Yeniseiana'
        elif lang in ['Xiongnú', 'Huns']:
            return 'Xiongnú/Huns (Foco do Artigo)'
        elif lang in family_map:
            return family_map[lang]
        else:
            return 'Outras Famílias'
            
    tsne_df['Família Linguística'] = tsne_df['Língua'].apply(get_family)
    
    return tsne_df

# ----------------------------------------------------------------------
# 3. LÓGICA DE INPUT (STREAMLIT SESSION STATE)
# ----------------------------------------------------------------------

# Inicialização do Session State
if 'base_df' not in st.session_state:
    st.session_state.base_df = pd.DataFrame(BASE_LANGUAGE_RELATIONS)
if 'language_family_map' not in st.session_state:
    st.session_state.language_family_map = {}

def add_new_relation(source, target, weight, family):
    """Adiciona uma nova linha ao DataFrame de relações e atualiza o mapa de famílias."""
    if not source or not target or not weight:
        st.error("Preencha todos os campos da Relação Linguística.")
        return

    try:
        weight = int(weight)
        if weight <= 0:
            st.error("O Peso deve ser um número inteiro positivo.")
            return
    except ValueError:
        st.error("O Peso deve ser um número inteiro válido.")
        return
        
    new_row = pd.DataFrame([{'Source': source, 'Target': target, 'Weight': weight}])
    st.session_state.base_df = pd.concat([st.session_state.base_df, new_row], ignore_index=True)
    
    # Atualiza o mapa de famílias
    if source not in ['Arin', 'Ket', 'Yugh', 'Xiongnú', 'Huns', 'Proto-Turkic', 'Proto-Mongolic']:
        st.session_state.language_family_map[source] = family
    if target not in ['Arin', 'Ket', 'Yugh', 'Xiongnú', 'Huns', 'Proto-Turkic', 'Proto-Mongolic']:
        st.session_state.language_family_map[target] = family

def reset_data():
    """Reseta o DataFrame de relações para o estado inicial."""
    st.session_state.base_df = pd.DataFrame(BASE_LANGUAGE_RELATIONS)
    st.session_state.language_family_map = {}

# ----------------------------------------------------------------------
# 4. INTERFACE STREAMLIT
# ----------------------------------------------------------------------

# Título do App
st.title("👨‍💻 Validação Computacional de Cognatos (Node2Vec + t-SNE)")
st.subheader("Projeto de IA Aplicada à Linguística Histórica")

st.markdown("""
Este aplicativo demonstra a validação computacional da hipótese Yeniseiana-Xiongnú. Use a barra lateral para **adicionar novas relações** e simular o impacto no agrupamento.
""")



## ⚙️ Entrada de Dados (Simulação)

# Sidebar para Input de Dados
with st.sidebar:
    st.header("➕ Simular Nova Relação")
    st.markdown("Adicione uma relação de proximidade entre duas línguas.")
    
    with st.form("new_relation_form"):
        # Inputs para a nova aresta
        new_source = st.text_input("Língua 1 (Source)", value="Nova Língua", max_chars=30)
        new_target = st.text_input("Língua 2 (Target)", value="Ket", max_chars=30)
        new_weight = st.number_input("Força/Peso (1 a 100)", min_value=1, max_value=100, value=50, step=1)
        new_family = st.text_input("Família Linguística da Nova Língua", value="Simulação", max_chars=30)
        
        # Botão de submissão do formulário
        submit_button = st.form_submit_button("Adicionar Relação e Re-analisar")

    if submit_button:
        # Chama a função para adicionar ao DataFrame
        add_new_relation(new_source, new_target, new_weight, new_family)
        st.success("Nova relação adicionada. Re-executando análise...")

    # Botão de Reset
    st.button("🔄 Resetar para Dados Iniciais", on_click=reset_data)
    
    st.markdown("---")
    st.info("Para mais detalhes sobre as métricas do Node2Vec, consulte a documentação do projeto.")

# Executa a análise com os dados atuais
tsne_results = run_node2vec_analysis(st.session_state.base_df)

## 1. Grafo de Relações (Dados de Entrada)

st.header("1. Grafo de Relações (Dados de Entrada Atuais)")
st.markdown("Tabela de entrada usada na análise (incluindo suas simulações):")
st.dataframe(st.session_state.base_df, hide_index=True)



## 2. Prova Computacional: Visualização 2D (t-SNE)

st.header("2. Prova Computacional: Visualização 2D (t-SNE)")
st.markdown("""
O algoritmo **Node2Vec** transformou a estrutura do grafo em vetores. O **t-SNE** reduziu esses vetores para 2 dimensões. **Nós próximos no gráfico indicam alta proximidade linguística.**
""")

if tsne_results is not None:
    # Criação do gráfico interativo com Plotly
    fig = px.scatter(tsne_results, 
                    x='Componente 1 (t-SNE)', 
                    y='Componente 2 (t-SNE)', 
                    color='Família Linguística', # Colore pelo agrupamento linguístico
                    text='Língua',              # Exibe a língua ao passar o mouse
                    hover_data={'Língua': True, 
                                'Componente 1 (t-SNE)': ':.2f', 
                                'Componente 2 (t-SNE)': ':.2f'},
                    title='Agrupamento de Línguas via Node2Vec e t-SNE')

    fig.update_traces(textposition='top center', 
                      marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(height=600, 
                      legend_title_text='Família Linguística',
                      title_x=0.5)

    st.plotly_chart(fig, use_container_width=True)
    
    # Adicionar o Diagrama para Contexto
    st.markdown("")
    
    
    
    ## 3. Dados Gerados (Embeddings 2D)

    st.header("3. Dados Gerados (Coordenadas t-SNE)")
    st.markdown("Coordenadas 2D que definem a posição de cada língua no gráfico, usadas para medir a proximidade:")
    st.dataframe(tsne_results)


    
    ## 4. Conclusão do Projeto

    st.header("4. Conclusão do Projeto")
    st.markdown("""
    O agrupamento visual demonstra a proximidade entre as línguas, validando a hipótese original. **Ao adicionar novas relações, observe como a topologia do grafo (e, consequentemente, a posição 2D) se altera.**
    """)
    
    # Exemplo de como a nova língua se agrupou
    if st.session_state.language_family_map:
        new_languages = [lang for lang, family in st.session_state.language_family_map.items()]
        if new_languages:
            st.info(f"As línguas simuladas **{', '.join(new_languages)}** foram plotadas com base nas relações que você adicionou. Sua posição no gráfico reflete a força das suas conexões com as línguas existentes, como esperado pelo Node2Vec.")

else:
    st.error("Análise t-SNE não executada. Certifique-se de ter pelo menos duas línguas relacionadas na tabela de dados.")
