import streamlit as st
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
# Importação atualizada para ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# --- Configurações Iniciais ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Nome do arquivo para armazenar o histórico de feedbacks
FEEDBACK_HISTORY_FILE = "feedback_history.json"

# Função para carregar o histórico de feedbacks
def load_feedback_history():
    if os.path.exists(FEEDBACK_HISTORY_FILE):
        with open(FEEDBACK_HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                st.warning("O arquivo de histórico de feedbacks está corrompido ou vazio. Criando um novo.")
                return []
    return []

# Função para salvar o histórico de feedbacks
def save_feedback_history(history):
    with open(FEEDBACK_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# Inicializar modelo e memória (reutilizamos para cada sessão do Streamlit, se possível)
@st.cache_resource
def initialize_llm_and_agent(api_key, model_name="gpt-4-turbo", temperature=0.7):
    if not api_key:
        st.error("Chave da API da OpenAI não encontrada. Por favor, adicione OPENAI_API_KEY no seu arquivo .env.")
        st.stop()
    try:
        llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            openai_api_key=api_key,
            # max_tokens=1024 # Opcional: Limita o tamanho da resposta do LLM para controlar custos
        )
        # A memória aqui é para a "conversa" atual da cadeia, não para o histórico de feedbacks
        memory = ConversationBufferMemory(return_messages=True)
        assistente = ConversationChain(llm=llm, memory=memory, verbose=False)
        return assistente
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo da OpenAI: {e}. Verifique sua chave da API e conexão.")
        st.stop()


# --- Interface Streamlit ---
st.set_page_config(
    page_title="Avaliador de Entrevistas AI",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Avaliador de Entrevistas com IA")
st.markdown("Faça o upload da transcrição da sua entrevista e receba um feedback detalhado sobre sua performance.")

st.sidebar.header("Configurações do Modelo")
model_name = st.sidebar.selectbox(
    "Escolha o Modelo OpenAI:",
    ("gpt-4-turbo", "gpt-3.5-turbo"),
    index=0 # gpt-4-turbo como padrão
)
temperature = st.sidebar.slider(
    "Criatividade (Temperatura):",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05
)

# Re-inicializa o assistente se as configurações mudarem
if 'current_assistente' not in st.session_state or \
   st.session_state.get('current_assistente_model') != model_name or \
   st.session_state.get('current_assistente_temp') != temperature:
    st.session_state.current_assistente = initialize_llm_and_agent(OPENAI_API_KEY, model_name, temperature)
    st.session_state.current_assistente_model = model_name
    st.session_state.current_assistente_temp = temperature


uploaded_file = st.file_uploader(
    "Selecione o arquivo da transcrição da entrevista (.txt)",
    type="txt",
    help="O arquivo deve conter a transcrição da sua entrevista. Tente formatar as falas do Recrutador e Candidato se possível, mas a IA tentará inferir."
)

feedback_placeholder = st.empty() # Placeholder para o feedback

if uploaded_file is not None:
    transcricao = uploaded_file.read().decode("utf-8")
    st.subheader("Pré-visualização da Transcrição:")
    st.text_area("Conteúdo da Transcrição", transcricao, height=200, disabled=True)

    if st.button("Gerar Feedback", key="generate_feedback_button"):
        if not OPENAI_API_KEY:
            st.error("Erro: A chave da API da OpenAI não foi carregada. Verifique seu arquivo .env.")
            st.stop()

        with st.spinner("Analisando a transcrição e gerando feedback... Isso pode levar alguns momentos."):
            # Carregar histórico de feedbacks para a IA
            historico_feedbacks = load_feedback_history()
            historico_str = "\n".join([
                f"--- Feedback {i+1} ---\nData: {item.get('data', 'N/A')}\nArquivo: {item.get('nome_arquivo', 'N/A')}\nNota: {item.get('nota', 'N/A')}\nResumo: {item.get('resumo', 'N/A')}\n" # Pega um resumo
                for i, item in enumerate(historico_feedbacks)
            ]) if historico_feedbacks else "Nenhum histórico de feedback anterior disponível."

            # --- INÍCIO DO PROMPT REFINADO PARA INTERFACE ---
            pergunta = f"""
            Você é um avaliador profissional e imparcial de entrevistas de emprego (técnicas e comportamentais). Sua missão é fornecer um feedback detalhado e construtivo **focando exclusivamente na performance do candidato (EU)**, com base em trechos reais da entrevista transcrita abaixo.

            **Instruções Cruciais para a Análise:**
            * A transcrição pode não ter identificação explícita de quem fala. Sua tarefa é **inferir quem é o candidato (EU)** com base nas perguntas típicas do recrutador e nas respostas que se alinham à uma apresentação pessoal ou profissional.
            * **Priorize a análise das MINHAS falas.** O feedback deve ser sobre a **MINHA comunicação, postura, clareza e estratégia de respostas**, e não sobre as perguntas do recrutador.
            * Ao citar trechos, **deixe claro se o trecho é uma pergunta do recrutador ou uma fala MINHA**, mas use-o apenas para contextualizar a **MINHA resposta ou a MINHA performance**.
            * Se o trecho for longo, cite apenas a parte mais relevante e adicione "..." se for truncado.
            * Certifique-se de que cada um dos 8 tópicos solicitados abaixo seja abordado de forma completa e detalhada, com exemplos.

            Sua resposta DEVE ser estruturada exatamente com os seguintes tópicos numerados, incluindo o número e o nome do tópico em negrito:

            1.  **Nota geral de 0 a 10 da MINHA performance.**
            2.  **Meus principais acertos (do candidato)**
            3.  **O que ME prejudicou (erros, falas inseguras, falta de clareza ou foco)**
            4.  **Sugira formas melhores de EU ME expressar**
            5.  **O que reorganizar no MEU roteiro de respostas**
            6.  **Evolução com base na memória de entrevistas anteriores**
            7.  **Dicas mentais e estratégias para melhorar a segurança e desempenho**
            8.  **Exemplos práticos de como responder melhor**

            Detalhes para cada tópico:

            **1. Nota geral de 0 a 10 da MINHA performance.**
                - Forneça uma nota numérica clara.

            **2. Meus principais acertos (do candidato)**
                - Com trechos específicos da transcrição que comprovem isso (ex: "Quando o candidato disse '...', demonstrou clareza/confiança/...").

            **3. O que ME prejudicou (erros, falas inseguras, falta de clareza ou foco)**
                - Com trechos reais **DAS MINHAS falas** que demonstrem os pontos fracos.

            **4. Sugira formas melhores de EU ME expressar**
                - Reescreva partes ruins **DAS MINHAS falas** de forma ideal, mostrando como eu poderia ter formulado a resposta.

            **5. O que reorganizar no MEU roteiro de respostas**
                - Temas que deveriam vir antes, respostas que se alongam sem necessidade etc.

            **6. Evolução com base na memória de entrevistas anteriores**
                - Use o seguinte histórico de feedbacks para a análise de evolução, regressão ou estagnação em aspectos específicos **DA MINHA performance**:
                Histórico de Feedbacks Anteriores:
                \"\"\"
                {historico_str}
                \"\"\"
                - Se o histórico estiver vazio ou não houver dados relevantes, indique isso e ofereça dicas gerais para a próxima.

            **7. Dicas mentais e estratégias para melhorar a segurança e desempenho**
                - Orientações práticas e acionáveis.

            **8. Exemplos práticos de como responder melhor**
                - Dê exemplos práticos de como EU poderia responder melhor, com trechos simulados que eu poderia usar no lugar do que foi dito.

            ⚠️ **IMPORTANTE:**
            -   Seja direto, detalhado e específico.
            -   Não resuma demais. Justifique com exemplos reais sempre que possível, **priorizando citações das MINHAS falas**.
            -   **Foço EXCLUSIVAMENTE na MINHA qualidade de comunicação, clareza, postura e estratégia como candidato.**
            -   Lembre-se: o objetivo é a MINHA evolução constante.

            Transcrição da entrevista:
            \"\"\"
            {transcricao}
            \"\"\"
            """
            # --- FIM DO PROMPT REFINADO PARA INTERFACE ---

            resposta_raw = st.session_state.current_assistente.predict(input=pergunta)

            # Tenta extrair a nota para o histórico
            # Ajuste a regex para ser mais flexível, caso a IA mude um pouco a frase
            nota_match = re.search(r"Nota geral de 0 a 10 da MINHA performance.*:?\s*(\d+(\.\d+)?)", resposta_raw, re.IGNORECASE | re.DOTALL)
            nota_final = float(nota_match.group(1)) if nota_match else "N/A"

            # Salvar o feedback no histórico
            new_feedback_entry = {
                "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "nome_arquivo": uploaded_file.name,
                "nota": nota_final,
                "resumo": resposta_raw[:500] + "..." if len(resposta_raw) > 500 else resposta_raw, # Salva um resumo
                "feedback_completo": resposta_raw
            }
            historico_feedbacks.append(new_feedback_entry)
            save_feedback_history(historico_feedbacks)

            feedback_placeholder.subheader("--- FEEDBACK DO ASSISTENTE ---")
            feedback_placeholder.markdown(resposta_raw)

            st.success("Feedback gerado e salvo no histórico!")

# Exibir histórico de feedbacks (opcional)
if st.sidebar.checkbox("Mostrar Histórico de Feedbacks Anteriores"):
    history = load_feedback_history()
    if history:
        st.sidebar.subheader("Histórico de Avaliações")
        for i, entry in enumerate(reversed(history)): # Mostra os mais recentes primeiro
            with st.sidebar.expander(f"Avaliação de {entry.get('data', 'N/A')} ({entry.get('nome_arquivo', 'N/A')}) - Nota: {entry.get('nota', 'N/A')}"):
                st.write(f"**Data:** {entry.get('data', 'N/A')}")
                st.write(f"**Arquivo:** {entry.get('nome_arquivo', 'N/A')}")
                st.write(f"**Nota:** {entry.get('nota', 'N/A')}")
                st.write(f"**Resumo:** {entry.get('resumo', 'N/A')}")
                if st.button(f"Ver Feedback Completo", key=f"full_feedback_{i}"):
                    feedback_placeholder.subheader(f"--- FEEDBACK COMPLETO DE {entry.get('data', 'N/A')} ---")
                    feedback_placeholder.markdown(entry.get('feedback_completo', 'Feedback não disponível.'))
                    st.toast("Feedback completo exibido na área principal.")
    else:
        st.sidebar.info("Nenhum feedback anterior salvo.")