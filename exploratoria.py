import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    livros = livros.rename(columns=colunas)
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
    return livros

df = carregar_csv()
df = df.drop_duplicates()

st.header('Dataframe geral dos Livros')
st.dataframe(df, use_container_width=True) # Ou pode usar st.write(df)
st.write('---')

st.header('Quantidade de valores Null')
st.write(df.isnull().sum())
st.write('---')

st.header('Contagem de aparições dos autores')
st.write(df['Autor(a)'].value_counts())
st.write('---')

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
st.write('---')

st.header('Contagem dos idiomas')
st.write(df['Idioma'].value_counts())
st.write('---')

st.header('Quantidade de avaliações feitas')
barra = px.bar(df, x = 'Avaliação', y = 'Ano')
st.plotly_chart(barra)
st.write('---')

st.header('Os 50 livros mais bem avaliados')
st.write(df.nlargest(50, 'Avaliação'))
st.write('---')

st.header('Relação entre números de páginas e avaliações (3D)')
scatter1, scatter2 = st.columns(2)
with scatter1:
    dispersao = px.scatter(df, x = 'Páginas', y = 'Avaliação')
    st.plotly_chart(dispersao)
with scatter2:
    dispersao_3d = px.scatter_3d(df, x = 'Páginas', y = 'Avaliação', z = 'Ano')
    st.plotly_chart(dispersao_3d)
st.write('---')

st.header('Quantidade de livros lançados por ano')
st.write(df.groupby('Ano').size())
livros_1990 = df[df['Ano'] >= 1990]
contagem_livros = livros_1990['Ano'].value_counts().sort_index()
linha = px.line(contagem_livros, x = contagem_livros.index, y = contagem_livros.values)
linha.update_layout(title = 'Lançados desde 1990',
                    xaxis_title = 'Ano',
                    yaxis_title = 'Quantidade de livros')
st.plotly_chart(linha)
st.write('---')

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
st.write('---')

st.header('Todos os livros por idioma')
selecionar_idioma = st.selectbox('Selecione um idioma:', df['Idioma'].unique())
dados_filtrados = df[df['Idioma'] == selecionar_idioma]
st.write(dados_filtrados)
st.write('---')
