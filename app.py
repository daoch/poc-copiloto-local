import time
from pathlib import Path

import pandas as pd
import streamlit as st

from logger import save_log
from rag import ask_copilot


st.set_page_config(
    page_title="POC Copiloto Empresarial Local",
    page_icon=":robot_face:",
    layout="wide",
)

st.title("POC Copiloto Empresarial Local")
st.write(
    "Demo local con documentos ficticios, RAG, ChromaDB dockerizado, embeddings locales y Ollama."
)

with st.sidebar:
    st.header("Configuracion")

    role = st.selectbox(
        "Rol del usuario",
        ["general", "rrhh", "ti", "ventas", "gerencia"],
    )

    st.markdown("---")
    st.write("Ejemplos de preguntas:")

    st.code("Cuantos dias de vacaciones corresponden despues de un ano?")
    st.code("No puedo conectarme a la VPN, que debo hacer?")
    st.code("Resume los KPIs de ventas del Q1 2026.")
    st.code("Cual sera el bono anual del proximo ano?")

tab_chat, tab_logs = st.tabs(["Chat", "Logs"])

with tab_chat:
    question = st.text_area(
        "Pregunta",
        placeholder="Ej: Cuantos dias de vacaciones corresponden despues de un ano?",
    )

    if st.button("Preguntar", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta.")
        else:
            start = time.time()

            with st.spinner("Buscando en ChromaDB y generando respuesta con Ollama..."):
                result = ask_copilot(question=question, role=role)

            latency = round(time.time() - start, 2)

            st.markdown(result["answer"])
            st.caption(f"Latencia: {latency} segundos")

            save_log(
                role=role,
                question=question,
                answer=result["answer"],
                latency=latency,
            )

            with st.expander("Ver chunks recuperados desde ChromaDB"):
                for source in result["sources"]:
                    st.write(f"**Fuente:** {source['file_name']}")
                    st.write(f"**Area:** {source['area']}")
                    st.write(source["text"])
                    st.markdown("---")

with tab_logs:
    log_file = Path("data/logs.csv")

    if log_file.exists():
        df = pd.read_csv(log_file)

        st.subheader("Metricas basicas")

        col1, col2 = st.columns(2)

        col1.metric("Consultas totales", len(df))
        col2.metric("Latencia promedio", round(df["latency"].mean(), 2))

        st.subheader("Logs")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Todavia no hay logs.")
