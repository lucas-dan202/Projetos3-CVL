import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st

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
def carregar_csv(): # Carregar e fazer alguns ajustes no dataset
    livros = pd.read_csv('dados.csv')
    colunas = {
        'titulo': 'Título',
        'autor': 'Autor(a)',
        'ISBN_13': 'ISBN_13',
        'ISBN_10': 'ISBN_10',
        'ano': 'Ano',
        'paginas': 'Páginas',
        'idioma': 'Idioma',
        'editora': 'Editora',
        'rating': 'Avaliação',
        'avaliacao': 'Quantidade de avaliações',
        'resenha': 'Quantidade de resenhas',
        'abandonos': 'Quantidade de abandonos',
        'relendo': 'Quantidade que estão relendo',
        'querem_ler': 'Quantidade que querem ler',
        'lendo': 'Quantidade que estão lendo',
        'leram': 'Quantidade que leram',
        'descricao': 'Descrição',
        'genero': 'Gênero',
        'male': 'Masculino (%)',
        'female': 'Feminino (%)'
        }
    # Renomeação das colunas
    livros = livros.rename(columns=colunas)
    # Como a coluna ISBN_13 é uma identificação, foi tranformado para string
    livros['ISBN_13'] = livros['ISBN_13'].astype(object)
    # Tratamento das avaliações maiores que 5, usando a média
    menores_5 = livros.loc[livros['Avaliação'] <= 5, 'Avaliação']
    media = round(menores_5.mean(), 1)
    livros['Avaliação'] = livros['Avaliação'].apply(lambda x: media if x > 5 else x)
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

st.header('Contagem dos idiomas')
st.write(df['Idioma'].value_counts())
st.write('---')

st.header('Quantidade de avaliações feitas')
barra = px.bar(df, x='Avaliação', y='Ano')
st.plotly_chart(barra)
st.write('---')

st.header('Os 50 livros mais bem avaliados')
st.write(df.nlargest(50, 'Avaliação'))
st.write('---')

st.header('Relação entre números de páginas e avaliações (3D)')
dispersao = px.scatter(df, x='Páginas', y='Avaliação')
st.plotly_chart(dispersao)
dispersao_3d = px.scatter_3d(df, x='Páginas', y='Avaliação', z='Ano')
st.plotly_chart(dispersao_3d)
st.write('---')

st.header('Quantidade de livros lançados por ano')
st.write(df.groupby('Ano').size())
st.write('---')

st.header('Todos os livros por idioma')
selecionar_idioma = st.selectbox('Selecione um idioma:', df['Idioma'].unique())
dados_filtrados = df[df['Idioma'] == selecionar_idioma]
st.write(dados_filtrados)
st.write('---')
