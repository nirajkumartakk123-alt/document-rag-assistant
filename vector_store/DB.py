from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)

result = vectorstore.similarity_search("What is used for Data Analysis?")

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain Deep Learning")

for d in docs:
    print(d.page_content)