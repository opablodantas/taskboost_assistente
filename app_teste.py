import os
import streamlit as st
from dotenv import load_dotenv
import warnings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore")
load_dotenv()

# Configuração inicial
if "tema" not in st.session_state:
    st.session_state.tema = "Claro"

st.set_page_config(
    page_title="TaskBoost - Assistente IA",
    page_icon="🤖",
    layout="wide",
)

# Estilo de acordo com o tema
if st.session_state.tema == "Escuro":
    st.markdown("""<style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 10px;
            margin: 8px 0;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
        .user-bubble {
            background-color: #2C3E50;
            border-left: 4px solid #3498DB;
        }
        .ai-bubble {
            background-color: #34495E;
            border-left: 4px solid #2ECC71;
        }
        .typing-indicator span {
            display: inline-block;
            width: 6px;
            height: 6px;
            margin: 0 2px;
            background: #ccc;
            border-radius: 50%;
            animation: typing 1s infinite;
        }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.2); }
        }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #e2e6ec);
            color: #000000;
        }
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 10px;
            margin: 8px 0;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        .user-bubble {
            background-color: #d0ebff;
            border-left: 4px solid #339af0;
        }
        .ai-bubble {
            background-color: #f1f3f5;
            border-left: 4px solid #66bb6a;
        }
        .titulo-personalizado {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .typing-indicator span {
            display: inline-block;
            width: 6px;
            height: 6px;
            margin: 0 2px;
            background: #888;
            border-radius: 50%;
            animation: typing 1s infinite;
        }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.2); }
        }
    </style>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("LOGO_TASKBOOST.png", width=150)
    st.markdown("## LIBERTE-SE DO TRABALHO REPETITIVO. FOQUE NO QUE IMPORTA")
    st.markdown("""### BEM-VINDO  
    Tire suas dúvidas sobre a nossa empresa aqui 😊""", unsafe_allow_html=True)

    st.markdown("**🔍 Geração de Relatórios**\nRelatórios automáticos em PDF.")
    st.markdown("**⚙️ Automação de Tarefas**\nSimplifique operações repetitivas.")

    tema = st.selectbox("🎨 TEMA", ["Claro", "Escuro"], index=0 if st.session_state.tema == "Claro" else 1)
    st.session_state.tema = tema

    if st.button("🔁 Reiniciar Conversa"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()

    st.markdown("---")

# Embeddings
embedding_model = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

# Carregar e indexar documentos PDF
@st.cache_resource
def carregar_index():
    loader = PyPDFDirectoryLoader("arquivos/")
    documentos = loader.load()
    return FAISS.from_documents(documentos, embedding_model)

index = carregar_index()

# Template do assistente
template = """
Você é o assistente virtual da TaskBoost, uma empresa especializada em automatização de tarefas e criação de relatórios para pequenos negócios.

Sua missão é ajudar os usuários a entender os serviços da empresa, responder dúvidas com clareza, empatia e sempre com um tom profissional e amigável.

Seja objetivo, mas converse de forma natural, como um humano prestativo falaria com um cliente curioso ou em dúvida.

Use emojis com moderação quando fizer sentido, e jamais invente informações. Seja honesto quando não souber algo com base nos documentos.
"""

# Memória de conversa
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# LLM e chain de conversa
llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.7)
conversation_chain = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Histórico de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Título principal
if st.session_state.tema == "Claro":
    st.markdown('<div class="titulo-personalizado">🤖 TaskBoost - Seu Assistente Virtual</div>', unsafe_allow_html=True)
else:
    st.title("🤖 TaskBoost - Seu Assistente Virtual")

# Entrada do usuário
pergunta = st.chat_input("Digite aqui...")

# Resposta com contexto via FAISS
def obter_resposta_com_contexto(pergunta):
    documentos_relacionados = index.similarity_search(pergunta, k=2)
    contexto = "\n".join([doc.page_content[:1000] for doc in documentos_relacionados])
    full_prompt = f"{template}\n\nDocumentos disponíveis:\n{contexto}\n\nUsuário: {pergunta}\nAssistente:"
    resposta = conversation_chain.run(input=full_prompt)
    return resposta

# Processamento da pergunta
if pergunta:
    with st.spinner("🤖 O assistente está pensando... aguarde"):
        try:
            resposta = obter_resposta_com_contexto(pergunta)
        except Exception as e:
            resposta = "❌ Ocorreu um erro ao tentar gerar a resposta. Por favor, tente novamente."
            st.error(str(e))
        st.session_state.chat_history.append(("usuário", pergunta))
        st.session_state.chat_history.append(("assistente", resposta))

# Exibição do histórico
st.markdown('<div style="max-height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
for autor, mensagem in st.session_state.chat_history:
    if autor == "usuário":
        st.markdown(f'<div class="chat-bubble user-bubble">🧑‍💼 {mensagem}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble ai-bubble">🤖 {mensagem}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
