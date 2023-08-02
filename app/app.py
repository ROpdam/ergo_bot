import chainlit as cl
from functions import format_response, qa_chain

# from prompt_testing import test_response


@cl.on_chat_start
async def init():
    """"""
    await cl.Message(content="Hi! I am the Ergolog chatbot how can I help?").send()

    cl.user_session.set("chain", qa_chain)


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    chain = cl.user_session.get("chain")
    response = await chain.acall(message)
    response = format_response(response)
    # response = format_response(test_response)

    await cl.Message(
        author="Retriever",
        content=response["article_names"],
        elements=response["article_links"],
        indent=1,
    ).send()
    await cl.Message(content=response["answer"], indent=0).send()
