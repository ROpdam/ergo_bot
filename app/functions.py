from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.fake import FakeListLLM
from langchain.vectorstores import Chroma
from prompt_testing import test_response

k = 4
db_persist_directory = "ergologs_data/db"
load_dotenv()

# OpenAI embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=db_persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": k})

qa_chain = RetrievalQA.from_chain_type(
    llm=FakeListLLM(
        responses=test_response
    ),  # ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)


def q_and_a(user_query: str) -> dict:
    """"""
    llm_response = qa_chain(user_query)

    return {
        "answer": llm_response["result"],
        "source_docs": llm_response["source_documents"],
    }
