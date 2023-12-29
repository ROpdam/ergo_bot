import chainlit as cl
from functions import create_qa_chain, format_response

# from prompt_testing import test_response


@cl.on_chat_start
async def init():
    """"""
    elements = [
        cl.Text(
            name="Example Questions",
            content="""
    Is ginger healthy? Why?
    \n---
    How to improve endurance?
    \n---
    What are health benefits of green tea and why?
    \n---
    Why do athletes eat bananas?
    \n---
    How to gain muscle mass?
    \n---
    """,
        )
    ]

    await cl.Message(
        content="""Hi! I am the [Ergolog](https://www.ergo-log.com/) assistant how can I help? (Example Questions)""",
        elements=elements,
    ).send()

    cl.user_session.set("chain", create_qa_chain())


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
