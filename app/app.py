import chainlit as cl
from functions import format_response, qa_chain
from prompt_testing import test_response


@cl.on_chat_start
async def init():
    """"""
    await cl.Message(content="Hi! I am the Ergolog chatbot how can I help?").send()

    cl.user_session.set("chain", qa_chain)


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    # chain = cl.user_session.get("chain")
    # cb = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True #, answer_prefix_tokens=["FINAL", "ANSWER"]
    # )
    # cb.answer_reached = True
    # res = await chain.acall(message, callbacks=[cb])
    res = format_response(test_response)

    # print(len(res['doc_names']), len(res['doc_contents']))

    await cl.Message(
        author="Retriever",
        content=res["article_names"],
        elements=res["article_links"],
        indent=1,
    ).send()
    await cl.Message(content=res["answer"], indent=0).send()
