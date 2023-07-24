import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Ajuste da página geral
st.set_page_config(
    page_title = 'Análise Exploratória',
    page_icon = ':books:',
    layout = 'wide',
    menu_items= {
        'Get help': "https://streamlit.io",
        'Report a Bug': "https://blog.streamlit.io",
        'About': "Este é um projeto estudantil feito por alunos da **UFRPE**."
        } # type: ignore
)

# Criação de um cabeçalho
st.markdown('''# **Análise exploratória**

Por `camila`, `lucas` e `vanderlane`
---
''', unsafe_allow_html=True)

@st.cache_data
def carregar_csv():
    def extrair_generos(texto):
        if isinstance(texto, str):
            # Substituição de vírgulas e transformação em título
            texto = texto.replace(',', ' / ')
            texto = texto.title()
            # Verifica se existe um gênero válido
            match = regex.search(texto)
            if match:
                # Obtém o índice do primeiro gênero encontrado
                # e desconsidera a descrição antes do primeiro gênero
                index = match.start()
                texto = texto[index:]
                generos = regex.findall(texto)
                # Remoção das duplicatas
                generos = list(dict.fromkeys(generos))
                return ' / '.join(generos)
            else:
                # Retorna vazia se não houver gênero válido
                return ''
        else:
            # Retorna vazia se o texto não for string
            return ''
    # Carregamento e realização de alguns ajustes no dataset
    livros = pd.read_csv('dados.csv')
    colunas = {'titulo': 'Título', 'autor': 'Autor(a)', 'ISBN_13': 'ISBN_13', 'ISBN_10': 'ISBN_10',
                'ano': 'Ano', 'paginas': 'Páginas', 'idioma': 'Idioma', 'editora': 'Editora',
                'rating': 'Avaliação', 'avaliacao': 'Quantidade de avaliações',
                'resenha': 'Quantidade de resenhas', 'abandonos': 'Quantidade de abandonos',
                'relendo': 'Quantidade que estão relendo', 'querem_ler': 'Quantidade que querem ler',
                'lendo': 'Quantidade que estão lendo', 'leram': 'Quantidade que leram',
                'descricao': 'Descrição', 'genero': 'Gênero', 'male': 'Masculino (%)', 'female': 'Feminino (%)'}
    # Renomeação das colunas
    livros = livros.rename(columns = colunas)
    # Tratamento da coluna ISBN_13 para string
    livros['ISBN_13'] = livros['ISBN_13'].astype(object)
    # Tratamento das avaliações maiores que 5, usando a média
    menores_5 = livros.loc[livros['Avaliação'] <= 5, 'Avaliação']
    media = round(menores_5.mean(), 1)
    livros['Avaliação'] = livros['Avaliação'].apply(lambda x: media if x > 5 else x)
    # Tratamento da coluna Gênero, excluindo as descições adicionais
    generos_validos = ['Ficção', 'Não-ficção', 'Ficção científica', 'Distopia', 'Crônicas', 'Poemas', 'Poesias',
                       'Fantasia', 'Aventura', 'Jogos', 'Esportes', 'Entretenimento', 'Humor', 'Comédia', 'Romance',
                       'Drama', 'Erótico', 'LGBT', 'GLS', 'Jovem adulto', 'Infantojuvenil', 'Infantil', 'Educação',
                       'Matemática', 'Sociologia', 'História', 'História Geral', 'História do Brasil',
                       'Medicina e Saúde', 'Biologia', 'Política', 'Negócios e Empreendedorismo', 'Economia',
                       'Finanças', 'Literatura Estrangeira', 'Literatura Brasileira', 'Crime', 'Romance policial',
                       'Suspense e Mistério', 'Horror', 'Terror', 'Autoajuda', 'Biografia', 'Autobiografia',
                       'Memórias', 'Religião e Espiritualidade', 'Ensaios', 'Música', 'HQ', 'Comics', 'Mangá',
                       'Chick-lit']
    regex = re.compile(r'\b(' + '|'.join(generos_validos) + r')\b')
    livros['Gênero'] = livros['Gênero'].apply(extrair_generos)
    # Tratamento de valores NaN em colunas numéricas
    colunas_numericas = ['Avaliação', 'Quantidade de avaliações', 'Quantidade de resenhas',
                         'Quantidade de abandonos', 'Quantidade que estão relendo', 'Quantidade que querem ler',
                         'Quantidade que estão lendo', 'Quantidade que leram', 'Páginas', 'Ano']
    livros[colunas_numericas] = livros[colunas_numericas].fillna(0)
    return livros

df = carregar_csv()
df = df.drop_duplicates()

# Análises
def dataframe_geral():
    st.header('Dataframe geral dos Livros')
    st.dataframe(df, use_container_width = True) # Ou pode usar st.write(df)

def valores_null():
    st.header('Quantidade de valores Null')
    st.write(df.isnull().sum())

def contagem_autores():
    st.header('Quantidade de autores')
    st.write(df['Autor(a)'].value_counts())

def distribuicao_generos():
    st.header('Distribuição dos gêneros')
    contagem_generos = df['Gênero'].str.split(' / ').explode().value_counts()
    total_livros = contagem_generos.sum()
    porcentagens = contagem_generos / total_livros * 100
    dados = pd.DataFrame({'Gênero': contagem_generos.index,
                        'Quantidade': contagem_generos.values,
                        'Porcentagem': porcentagens.values})
    pizza = px.pie(dados,
                    values = 'Quantidade',
                    names = 'Gênero',
                    labels = {'Quantidade': 'Quantidade de Livros'})
    st.plotly_chart(pizza)

def contagem_idiomas():
    st.header('Contagem dos idiomas')
    st.write(df['Idioma'].value_counts())

def avaliacoes_feitas():
    st.header('Quantidade de avaliações feitas')
    barra = px.bar(df, x = 'Avaliação', y = 'Ano')
    st.plotly_chart(barra)

def mais_avaliados():
    st.header('Os 50 livros mais bem avaliados')
    st.write(df.nlargest(50, 'Avaliação'))

def paginas_e_avaliacoes():
    st.header('Relação entre números de páginas e avaliações (3D)')
    dispersao = px.scatter(df, x = 'Páginas', y = 'Avaliação')
    st.plotly_chart(dispersao)
    dispersao_3d = px.scatter_3d(df, x = 'Páginas', y = 'Avaliação', z = 'Ano')
    st.plotly_chart(dispersao_3d)

def livros_por_ano():
    st.header('Quantidade de livros lançados por ano')
    st.write(df.groupby('Ano').size())
    livros_1990 = df[df['Ano'] >= 1990]
    contagem_livros = livros_1990['Ano'].value_counts().sort_index()
    linha = px.line(contagem_livros, x = contagem_livros.index, y = contagem_livros.values)
    linha.update_layout(title = 'Lançados desde 1990',
                        xaxis_title = 'Ano',
                        yaxis_title = 'Quantidade de livros')
    st.plotly_chart(linha)

def variaveis_numericas():
    st.header('Correlação entre variáveis numéricas')
    colunas_numericas = ['Quantidade de avaliações',
                        'Quantidade de resenhas', 'Quantidade de abandonos',
                        'Quantidade que estão relendo', 'Quantidade que querem ler',
                        'Quantidade que estão lendo', 'Quantidade que leram']
    correlacao = df[colunas_numericas].corr()
    calor = go.Heatmap(z=correlacao.values,
                        x=correlacao.columns,
                        y=correlacao.columns,
                        colorscale='Viridis')
    mapa = go.Figure(data = [calor])
    st.plotly_chart(mapa)

def livros_por_idioma():
    st.header('Todos os livros por idioma')
    selecionar_idioma = st.selectbox('Selecione um idioma:', df['Idioma'].unique())
    dados_filtrados = df[df['Idioma'] == selecionar_idioma]
    st.write(dados_filtrados)

st.sidebar.title('Análises')
analises = ['📚 Dataframe geral dos Livros', '❓ Quantidade de valores Null', '👨‍💼 Quantidade de autores',
            '🎭 Gráfico pizza', '🗣️ Contagem dos idiomas', '⭐️ Gráfico barra',
            '🏆 Os 50 livros mais bem avaliados', '📊 Gráfico de dispersão (3D)',
            '📅 Quantidade de livros lançados por ano', '🔗 Mapa de calor',
            '🗂️ Todos os livros por idioma', '🔢 Aplicação do algoritmo K-means']
pagina_escolhida = st.sidebar.selectbox('Selecione uma análise:', analises)

def kmeans_clustering():
    df_copia = df.copy()
    # Armazenar os títulos em uma variável
    titulos = df_copia['Título']
    # Transformação das colunas categóricas para aplicar o algoritmo de ML
    genero_dummy = df_copia['Gênero'].str.get_dummies(sep = ' / ')
    autor_dummy = pd.get_dummies(df_copia['Autor(a)'], prefix = 'Autor(a)')
    idioma_dummy = pd.get_dummies(df_copia['Idioma'], prefix = 'Idioma')
    editora_dummy = pd.get_dummies(df_copia['Editora'], prefix = 'Editora')
    df_copia = pd.concat([df_copia, genero_dummy, autor_dummy, idioma_dummy, editora_dummy], axis = 1)
    df_copia.drop(['Título', 'ISBN_13', 'ISBN_10', 'Gênero', 'Autor(a)', 'Idioma', 'Editora', 'Descrição'], axis = 1, inplace = True)
    df_copia.fillna(0, inplace = True)

    st.header('Aplicação do algoritmo K-means')

    # Determinar o número ideal de clusters usando o método do cotovelo (elbow method)
    num_clusters_range = range(1, 11)
    inercias = []
    for k in num_clusters_range:
        kmeans = KMeans(n_clusters = k, random_state = 0)
        kmeans.fit(df_copia)
        inercias.append(kmeans.inertia_)
    fig_cotovelo = go.Figure(data = go.Scatter(x = list(num_clusters_range), y = inercias, mode = 'lines+markers'))
    fig_cotovelo.update_layout(title = 'Método do Cotovelo para determinação do número de clusters',
                               xaxis_title = 'Número de clsuters',
                               yaxis_title = 'Inércia')
    st.plotly_chart(fig_cotovelo)

    # Escolher o número ideal de clusters com base no gráfico do cotovelo
    num_clusters = st.slider('Escolha o número de clusters:', 2, 10, 5)

    
    colunas_numericas = ['Avaliação', 'Quantidade de avaliações', 'Quantidade de resenhas',
                         'Quantidade de abandonos', 'Quantidade que estão relendo', 'Quantidade que querem ler',
                         'Quantidade que estão lendo', 'Quantidade que leram', 'Páginas', 'Ano']
    X = df_copia[colunas_numericas].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters = num_clusters, random_state = 0)
    kmeans.fit(X_scaled)
    clusters = kmeans.predict(X_scaled)

    df_copia['Título'] = titulos
    df_copia['Cluster'] = clusters

    st.dataframe(df_copia[['Título', 'Cluster']], use_container_width = True)

    fig_scatter = px.scatter(df_copia, x = 'Avaliação', y = 'Quantidade de avaliações', color = 'Cluster', hover_data = ['Título'])
    st.plotly_chart(fig_scatter)

if pagina_escolhida == '📚 Dataframe geral dos Livros':
   dataframe_geral()
if pagina_escolhida == '❓ Quantidade de valores Null':
    valores_null()
elif pagina_escolhida == '👨‍💼 Quantidade de autores':
    contagem_autores()
elif pagina_escolhida == '🎭 Gráfico pizza':
    distribuicao_generos()
elif pagina_escolhida == '🗣️ Contagem dos idiomas':
    contagem_idiomas()
elif pagina_escolhida == '⭐️ Gráfico barra':
    avaliacoes_feitas()
elif pagina_escolhida == '🏆 Os 50 livros mais bem avaliados':
    mais_avaliados()
elif pagina_escolhida == '📊 Gráfico de dispersão (3D)':
    paginas_e_avaliacoes()
elif pagina_escolhida == '📅 Quantidade de livros lançados por ano':
    livros_por_ano()
elif pagina_escolhida == '🔗 Mapa de calor':
    variaveis_numericas()
elif pagina_escolhida == '🗂️ Todos os livros por idioma':
    livros_por_idioma()
elif pagina_escolhida == '🔢 Aplicação do algoritmo K-means':
    kmeans_clustering()