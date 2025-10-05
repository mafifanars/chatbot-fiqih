# chatbot_fiqih_app.py — RAG → Fallback (versi: Top-K & threshold internal, spinner auto-hide)

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

# ====== CONFIG VECTORSTORE ======
VECTOR_DIR = "vectorstore"
INDEX_NAME = "fiqih_faiss"
EMBED_MODEL = "models/text-embedding-004"

# ====== INTERNAL RAG PARAMS ======
TOP_K = 3
RELEVANCE_THR = 0.67

st.set_page_config(page_title="Chatbot Fiqih - Belajar Fiqih dengan AI", page_icon="☪️")
st.title("☪️ Chatbot Fiqih")
st.caption("Sebuah chatbot fiqih yang simple dan sederhana (RAG → fallback).")

with st.sidebar:
    st.subheader("Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    allow_fallback = st.checkbox("Izinkan fallback ke pengetahuan umum", value=True)
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=(
                "Kamu adalah asisten fiqih berbahasa Indonesia yang menjawab dengan nada santai dan ramah, "
                "namun tetap sopan dan berbobot. Jawab ringkas (2–3 kalimat). "
                "Kalau pertanyaannya di luar fiqih, katakan singkat bahwa itu di luar bahasan fiqih."
            ),
        )
        st.session_state.llm_direct = llm
        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)
        st.session_state.pop("retriever", None)
        st.session_state.pop("vs", None)
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

def load_vectorstore():
    if "vs" in st.session_state:
        return st.session_state.vs
    try:
        emb = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=st.session_state._last_key
        )
        vs = FAISS.load_local(
            VECTOR_DIR, emb, index_name=INDEX_NAME, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Gagal memuat vectorstore: {e}")
        st.stop()
    st.session_state.vs = vs
    return vs

def get_retriever():
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": TOP_K})

if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("llm_direct", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("vs", None)
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

SYSTEM_RAG = """
Kamu adalah asisten fiqih AI yang ahli dalam Fiqih dan ramah.
Gunakan kutipan teks dari kitab berikut sebagai rujukan UTAMA dan SUMBER KEBENARAN untuk menjawab pertanyaan.
Jawablah pertanyaan pengguna secara lengkap dan jelas menggunakan informasi dari konteks yang diberikan.
Jika konteks hanya berisi daftar, jelaskan setiap poin dalam daftar tersebut secara ringkas.
Sintesis dan rangkum informasi dari beberapa sumber jika diperlukan untuk memberikan jawaban yang komprehensif.
Jika konteks yang ditemukan relevan, jelaskan isinya secukupnya untuk menjawab pertanyaan.

Pertanyaan: {question}

Konteks dari Kitab:
{context}
"""
PROMPT_RAG = ChatPromptTemplate.from_template(SYSTEM_RAG)

prompt = st.chat_input("Ketik pesan kamu disini...")

def score_to_similarity(score: float) -> float:
    try:
        return 1.0 / (1.0 + float(score))  # konversi jarak → ~similarity [0..1]
    except Exception:
        return 0.0

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RETRIEVE
    try:
        vs = load_vectorstore()
        retriever = get_retriever()
        try:
            results = vs.similarity_search_with_score(prompt, k=TOP_K)
            docs = [d for d, _ in results]
            sims = [score_to_similarity(s) for _, s in results]
            max_sim = max(sims) if sims else 0.0
        except Exception:
            docs = retriever.get_relevant_documents(prompt)
            max_sim = 1.0 if docs else 0.0

        def fmt(d):
            src = d.metadata.get("source", "fiqih.pdf")
            page = d.metadata.get("page", "?")
            return f"- ({src} p.{page}) {d.page_content}"

        context_block = "\n".join(fmt(d) for d in docs)
        need_fallback = (max_sim < RELEVANCE_THR) or (not docs)
        use_fallback = need_fallback and allow_fallback
    except Exception:
        need_fallback = True
        use_fallback = allow_fallback
        context_block = ""
        max_sim = 0.0

    # JAWAB
    try:
        if not use_fallback:
            final_prompt = PROMPT_RAG.format(question=prompt, context=context_block)
            with st.chat_message("assistant"):
                # spinner otomatis hilang saat selesai
                with st.spinner("✨ Model sedang menyiapkan jawaban..."):
                    answer = st.session_state.llm_direct.invoke(final_prompt).content
                    st.markdown(answer)

                # evidensi (jika ada)
                if context_block.strip():
                    with st.expander(f"📎 Sumber Jawaban (TOP_K={TOP_K}, max_sim≈{max_sim:.2f})"):
                        # Buat daftar sumber yang unik untuk menghindari duplikasi
                        unique_sources = set()
                        for d in docs:
                            # Ambil nama file dari path lengkap
                            source_file = os.path.basename(d.metadata.get("source", "N/A"))
                            # Ambil nomor halaman (ingat, page di metadata 0-indexed)
                            page_number = d.metadata.get("page", -1) + 1
                            if page_number > 0:
                                unique_sources.add(f"{source_file}, Halaman {page_number}")

                        # Tampilkan setiap sumber yang unik dalam format markdown list
                        for source in sorted(list(unique_sources)):
                            st.markdown(f"- {source}")

            mode_note = f"Mode: **RAG**{' (konteks kosong)' if (need_fallback and not allow_fallback) else ''}"
        else:
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]
            with st.chat_message("assistant"):
                with st.spinner("✨ Model sedang menyiapkan jawaban..."):
                    resp = st.session_state.agent.invoke({"messages": messages})
                    answer = resp["messages"][-1].content if "messages" in resp and resp["messages"] else \
                             "I'm sorry, I couldn't generate a response."
                    st.markdown(answer)
            mode_note = "Mode: **Fallback** (pengetahuan umum LLM, tidak memakai PDF)"
    except Exception as e:
        answer = f"An error occurred: {e}"
        mode_note = "Mode: error"
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.caption(mode_note)
    st.session_state.messages.append({"role": "assistant", "content": answer})