import streamlit as st
import raghelper

st.set_page_config(page_title="LangChain with RAG", layout="wide")
st.title("🌐LangChain with RAG: URL")
st.divider()

col_input, col_rag, col_normal = st.columns([1, 2, 2])

with col_input:
    target_url = st.text_input(label="Enter the Web Address to Process:")
    st.divider()
    prompt = st.text_input(label="Enter Your Question:", key="url_prompt")
    st.divider()
    submit_btn = st.button(label="Ask", key="url_button")
    st.divider()

    if submit_btn:
        with col_rag:
            with st.spinner("Preparing Response..."):
                st.success("RESPONSE - RAG Enabled")
                st.markdown(raghelper.rag_with_url(target_url=target_url, prompt=prompt))
                st.divider()

        with col_normal:
            with st.spinner("Preparing Response..."):
                st.info("RESPONSE - RAG Disabled")
                st.markdown(raghelper.ask_openai(prompt=prompt))
                st.divider()

st.title("📄LangChain with RAG: PDF")
st.divider()

col_input, col_rag, col_normal = st.columns([1, 2, 2])

with col_input:
    selected_file = st.file_uploader(label="Select the File to Process", type=["pdf"])
    st.divider()
    prompt = st.text_input(label="Enter Your Question:", key="pdf_prompt")
    st.divider()
    submit_btn = st.button(label="Ask", key="pdf_button")
    st.divider()

if submit_btn:

    with col_rag:
        with st.spinner("Preparing Response..."):
            st.success("RESPONSE - RAG Enabled")
            AI_Response, relevant_documents = raghelper.rag_with_pdf(filepath=f"./data/{selected_file.name}",
                                                                     prompt=prompt)
            st.markdown(AI_Response)
            st.divider()
            for doc in relevant_documents:
                st.caption(doc.page_content)
                st.markdown(f"Source: {doc.metadata}")
                st.divider()

        with col_normal:
            with st.spinner("Preparing Response..."):
                st.info("RESPONSE - RAG Disabled")
                st.markdown(raghelper.ask_openai(prompt=prompt))
                st.divider()


