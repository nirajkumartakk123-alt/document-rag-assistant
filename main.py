from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings()
vectorstore = Chroma(
    persist_directory = "chroma_db",
    embedding_function = embedding_model
)
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
llm = ChatMistralAI(model = "mistral-small-2603")

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpful AI assistant.
         Use only the provided context to answer the questions.
         if the answer is not present in the context,
         say: "I could not find the answer in the document."
        """),
        ("human",
         """Context:
         {context}
         Questions:
         {questions}
         """
         )
    ]
)

print("RAG system created")

print("press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = prompt.invoke({
        "context": context,
        "questions": query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")