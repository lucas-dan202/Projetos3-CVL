import pandas as pd
import matplotlib.pyplot as plt
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
    livros = livros.rename(columns=colunas) # Renomeação das colunas
    livros['ISBN_13'] = livros['ISBN_13'].astype(object) # Como a coluna ISBN_13 é uma identificação, foi tranformado para string
    return livros
df = carregar_csv()

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
plt.hist(df['Avaliação'])
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False) # Se não tiver essa linha, aparecerá um aviso que esse método será descontinuado
st.write('---')

st.header('Os 50 livros mais bem avaliados')
st.write(df.nlargest(50, 'Avaliação'))
st.write('---')

st.header('Relação entre números de páginas e avaliações')
plt.scatter(df['Páginas'], df['Avaliação'])
st.pyplot()
st.write('---')

st.header('Quantidade de livros lançados por ano')
st.write(df.groupby('Ano').size())
st.write('---')

st.header('Todos os livros por idioma')
selecionar_idioma = st.selectbox('Selecione um idioma:', df['Idioma'].unique())
dados_filtrados = df[df['Idioma'] == selecionar_idioma]
st.write(dados_filtrados)
st.write('---')
