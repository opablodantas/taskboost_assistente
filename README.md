
# 🤖 Assistente Virtual - TaskBoost

Este projeto é um assistente virtual inteligente para a TaskBoost, criado com **Streamlit**, **LangChain** e **OpenAI**, com capacidade de responder perguntas com base em documentos internos da empresa (em PDF). O objetivo é oferecer uma interface amigável e funcional que ajude clientes ou visitantes a tirar dúvidas sobre a empresa e seus serviços.

---

## 🚀 Funcionalidades

- Interface interativa via Streamlit com suporte a temas claro e escuro
- Respostas contextuais baseadas em documentos PDF (suporte via embeddings)
- Busca por similaridade com armazenamento persistente usando **ChromaDB**
- Processamento de linguagem natural com o modelo da **OpenAI**
- Histórico de conversa preservado durante a sessão

---

## 🛠️ Tecnologias Utilizadas

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Chroma Vector Store](https://www.trychroma.com/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 📁 Estrutura de Pastas



📦 assistente-taskboost/
├── arquivos/                # PDFs com informações sobre a empresa
├── db\_chroma/               # Vetor de embeddings persistente
├── LOGO\_TASKBOOST.png       # Logotipo exibido na sidebar
├── .env                     # Variáveis de ambiente (não subir no GitHub!)
├── requirements.txt         # Dependências do projeto
└── app.py                   # Código principal da aplicação

````

---

## 📄 Como Rodar Localmente

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/assistente-taskboost.git
   cd assistente-taskboost
````

2. Crie um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente no arquivo `.env`:

   ```
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

5. Adicione seus documentos PDF à pasta `arquivos/`.

6. Rode o aplicativo:

   ```bash
   streamlit run app.py
   ```

---

## 🧠 Observações

* A pasta `db_chroma` será criada automaticamente após o primeiro carregamento de PDFs e armazenará os vetores de embeddings localmente.
* Use documentos bem formatados e claros para obter melhores respostas do assistente.

---

## 📜 Licença

Este projeto está licenciado sob a **Licença MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

