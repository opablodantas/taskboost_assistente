import os
import streamlit as st
from dotenv import load_dotenv
import warnings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")
load_dotenv()

# Verificando se o tema já foi definido na sessão
if "tema" not in st.session_state:
    st.session_state.tema = "Claro"

st.set_page_config(
    page_title="TaskBoost - Assistente IA",
    page_icon="🤖",
    layout="wide",
)

# Estilo baseado no tema
if st.session_state.tema == "Escuro":
    st.markdown("""<style>
        .stApp { background-color: #000000; color: #FFFFFF; }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 12px;
            margin: 5px 0;
        }
        .user-bubble { background-color: #112137; color: #FFFFFF; }
        .ai-bubble { background-color: #112137; color: #FFFFFF; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        .stApp { background-color: #FFFFFF; color: #000000; }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 12px;
            margin: 5px 0;
        }
        .user-bubble { background-color: #cce5ff; color: #000000; }
        .ai-bubble { background-color: #f0f2f6; color: #000000; }
        .titulo-personalizado {
            font-size: 36px;
            font-weight: bold;
            color: #000000;
            margin-bottom: 20px;
        }
    </style>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("LOGO_TASKBOOST.png", width=150)
    st.markdown("## LIBERTE-SE DO TRABALHO REPETITIVO. FOQUE NO QUE IMPORTA")
    st.markdown("""
    ### BEM-VINDO  
    Tire suas dúvidas sobre a nossa empresa aqui 😊
    """, unsafe_allow_html=True)

    tema = st.selectbox("🎨 TEMA", ["Claro", "Escuro"], index=0 if st.session_state.tema == "Claro" else 1)
    st.session_state.tema = tema
    st.markdown("---")

# Embeddings
embedding_model = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

# Carregamento dos PDFs e criação do índice em memória
@st.cache_resource
def carregar_index():
    loader = PyPDFDirectoryLoader("arquivos/")
    documentos = loader.load()
    return Chroma.from_documents(documentos, embedding_model)

index = carregar_index()

# Template do assistente
template = """
Você é o assistente virtual da TaskBoost, uma empresa especializada em automatização de tarefas e criação de relatórios para pequenos negócios.

Sua missão é ajudar os usuários a entender os serviços da empresa, responder dúvidas com clareza, empatia e sempre com um tom profissional e amigável.

Seja objetivo, mas converse de forma natural, como um humano prestativo falaria com um cliente curioso ou em dúvida.

Use emojis com moderação quando fizer sentido, e jamais invente informações. Seja honesto quando não souber algo com base nos documentos.

Documentos disponíveis: {context}
Pergunta do usuário: {question}
Resposta do assistente:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# LLM
llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.7)
chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Função de resposta
def obter_resposta(pergunta):
    docs_relacionados = index.similarity_search(pergunta, k=5)
    return chain.run(input_documents=docs_relacionados, question=pergunta)

# Histórico da conversa
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Exibição do título
if st.session_state.tema == "Claro":
    st.markdown('<div class="titulo-personalizado">🤖 TaskBoost - Seu Assistente Virtual</div>', unsafe_allow_html=True)
else:
    st.title("🤖 TaskBoost - Seu Assistente Virtual")

# Entrada do usuário
pergunta = st.chat_input("Digite aqui...")

if pergunta:
    with st.spinner("Pensando..."):
        resposta = obter_resposta(pergunta)
        st.session_state.chat_history.append(("usuário", pergunta))
        st.session_state.chat_history.append(("assistente", resposta))

# Exibição da conversa
for autor, mensagem in st.session_state.chat_history:
    if autor == "usuário":
        st.markdown(f'<div class="chat-bubble user-bubble">🧑‍💼 {mensagem}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble ai-bubble">🤖 {mensagem}</div>', unsafe_allow_html=True)
