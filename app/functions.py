from typing import Union

import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import Chroma

system_template = """You are the ergolog chatbot. Use the
following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know,
don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of
the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

k = 4
db_persist_directory = "ergologs_data/db"
load_dotenv()

# OpenAI embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=db_persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": k})

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0, streaming=True, model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever,
)

# RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
#     # FakeListLLM(
#     #     responses=test_response
#     # ),
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     verbose=True,
#     chain_type_kwargs=chain_type_kwargs
# )


def get_doc_content(docs: list) -> Union[list, list]:
    """"""
    doc_names = "Sources: "
    doc_contents = []
    for idx, doc in enumerate(docs):
        doc_name = f" Document {idx+1} "
        doc_names += doc_name
        content = doc.page_content + "\nlink:" + doc.metadata["article_link"] + "\n"

        doc_contents.append(cl.Text(content=content, name=doc_name))

    return doc_names, doc_contents


def get_article_links(docs: list) -> Union[list, list]:
    """"""
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
    """"""
    # llm_response = qa_chain(user_query)
    # doc_names, doc_contents = get_doc_content(llm_response["source_documents"])
    article_links, article_names = get_article_links(llm_response["source_documents"])

    # return {"answer": llm_response["result"],
    #         "doc_names": doc_names,
    #         "doc_contents": doc_contents}
    return {
        "answer": llm_response["result"],
        "article_links": article_links,
        "article_names": article_names,
    }
