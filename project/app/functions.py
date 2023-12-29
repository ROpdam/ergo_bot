import os
from typing import Union

import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

prompt_template = """Use the following pieces of context to answer
the question at the end. If you don't know the answer, just
say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer using bullet points:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
k = 5

load_dotenv()
db_location = os.getenv("DB_LOCATION")

# OpenAI embeddings
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = OpenAIEmbeddings()

vectordb = FAISS.load_local(db_location, embedding_function)
retriever = vectordb.as_retriever(search_kwargs={"k": k})


def create_qa_chain() -> RetrievalQA:
    """
    Creates a question-answering (QA) chain using the RetrievalQA framework.

    This function initializes a RetrievalQA object with specified parameters.
    It uses the ChatOpenAI model (specifically gpt-3.5-turbo) as its language model (llm) and
    specifies a custom chain type for processing.

    Returns:
        RetrievalQA: An instance of the RetrievalQA class configured with the specified parameters.
    """
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
        chain_type_kwargs=chain_type_kwargs,
    )


def get_article_links(docs: list) -> Union[list, list]:
    """
    Formats a list of document objects into markdown-style article links.

    Args:
        docs (list): Documents with 'article_title' and 'article_link' in their metadata.

    Returns:
        tuple: Contains a formatted list of articles as cl.Text and their collective title.
    """
    article_names = "Ergolog Articles"
    content = ""
    for idx, doc in enumerate(docs):
        link_name = f" Article {idx+1} "
        content += (
            "\n"
            + "{}\n[{}]({})".format(
                doc.metadata["article_title"], link_name, doc.metadata["article_link"]
            )
            + "\n"
        )

    return [cl.Text(name=article_names, content=content)], article_names


def format_response(llm_response: dict) -> dict:
    """
    Transforms a language model's response into a dictionary with structured data.

    Args:
        llm_response (dict): Response from a language model, including 'result' and 'source_documents'.

    Returns:
        dict: Dictionary with 'answer' from the LLM, and 'article_links' and
        'article_names' extracted from source documents.
    """
    article_links, article_names = get_article_links(llm_response["source_documents"])

    return {
        "answer": llm_response["result"],
        "article_links": article_links,
        "article_names": article_names,
    }
