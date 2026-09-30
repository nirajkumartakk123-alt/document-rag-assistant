from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document_loaders/ALgorithms.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()

vectorstores = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)