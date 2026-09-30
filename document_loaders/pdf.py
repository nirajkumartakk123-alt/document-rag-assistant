from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document_loaders/analysis.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 1,
)

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()