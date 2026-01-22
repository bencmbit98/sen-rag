import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------
st.set_page_config(page_title="SEN RAG App", layout="wide")
st.title("📄 SEN RAG Assistant")

# Load API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --------------------------------------------------
# Build RAG pipeline (cached)
# --------------------------------------------------
@st.cache_resource
def build_qa_chain():
    documents = []

    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"docs/{file}")
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain

qa_chain = build_qa_chain()

# --------------------------------------------------
# UI
# --------------------------------------------------
query = st.text_input("Ask a question about SEN:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)

        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write(doc.metadata["source"])



