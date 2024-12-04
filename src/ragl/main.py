import streamlit as st
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add necessary imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import langchain

langchain.debug = True


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


loader = PyPDFDirectoryLoader("data/")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)

vector_store.add_documents(docs)

llm = ChatOllama(model="qwen2.5:14b")

rag_prompt = hub.pull("rlm/rag-prompt")

qa_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

st.sidebar.title("Options")
game = st.sidebar.selectbox("Choose a game", ("Imperial Struggle", "Kemet Blood and Sand", "Dune Imperium Uprising"))

st.title("ragl")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = qa_chain.stream(f"In the game {game}. {prompt}")
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
