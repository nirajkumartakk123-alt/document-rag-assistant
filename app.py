import os
import shutil
import streamlit as st

from dotenv import load_dotenv

import rag_core


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Cached model loaders (Streamlit-specific caching wraps the shared
# rag_core functions, so the models are loaded once per session, not
# once per query)
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return rag_core.get_embedding_model()


@st.cache_resource
def get_llm():
    return rag_core.get_llm()


def process_uploaded_files(uploaded_files):

    rag_core.clear_directory(rag_core.UPLOAD_DIR)

    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(rag_core.UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    embedding_model = get_embedding_model()

    vectorstore, chunk_count = rag_core.process_documents(
        file_paths, embedding_model
    )

    return vectorstore, chunk_count


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📚 Document RAG Assistant")

st.write(
    "Upload your PDF documents and ask questions about them."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🔄 Process Documents",
        use_container_width=True
    )

    st.divider()

    st.markdown(
        """
        ### How it works

        1. Upload PDF documents
        2. Click **Process Documents**
        3. Documents are split into chunks
        4. Mistral creates embeddings
        5. Hybrid search (BM25 + vector) + reranking retrieves context
        6. Ask questions about your documents
        """
    )


# --------------------------------------------------
# Process Documents
# --------------------------------------------------

if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF document."
        )

    else:

        with st.spinner(
            "Processing documents... This may take a moment."
        ):

            try:

                vectorstore, chunk_count = process_uploaded_files(
                    uploaded_files
                )

                st.session_state["documents_processed"] = True

                st.session_state["document_names"] = [
                    file.name for file in uploaded_files
                ]

                st.success(
                    f"Successfully processed "
                    f"{len(uploaded_files)} document(s) "
                    f"into {chunk_count} chunks."
                )

            except Exception as e:

                st.error(
                    f"Error while processing documents: {e}"
                )


# --------------------------------------------------
# Document Status
# --------------------------------------------------

if st.session_state.get("documents_processed", False):

    st.success("🟢 Documents are ready for questions.")

    with st.expander("📑 Processed Documents"):

        for name in st.session_state["document_names"]:

            st.write(f"• {name}")


# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions")

query = st.chat_input(
    "Ask something about your documents..."
)


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if query:

    if not os.path.exists(rag_core.CHROMA_DIR):

        st.warning(
            "Please upload and process documents before asking questions."
        )

    else:

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                try:

                    embedding_model = get_embedding_model()
                    llm = get_llm()

                    vectorstore = rag_core.get_vectorstore(embedding_model)

                    retriever = rag_core.build_retriever(
                        vectorstore, llm, k=10, top_n=5
                    )

                    answer, docs = rag_core.answer_question(
                        query, retriever, llm
                    )

                    st.write(answer)

                    with st.expander("🔍 View Retrieved Context"):

                        for i, doc in enumerate(docs):

                            st.markdown(f"**Source {i + 1}**")
                            st.write(doc.page_content)

                            if doc.metadata:
                                st.caption(
                                    f"Page: "
                                    f"{doc.metadata.get('page', 'Unknown') + 1}"
                                )

                            st.divider()

                except Exception as e:

                    st.error(
                        f"Error while generating answer: {e}"
                    )