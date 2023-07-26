from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

k = 4
db_persist_directory = "ergologs_data/db"

# OpenAI embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=db_persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": k})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)


def q_and_a(user_query: str) -> dict:
    """"""
    pass
